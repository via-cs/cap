import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler


def _init_dim(path):
    """
    Initializes input/output dimensions and max sequence length based on the dataset.
    Args:
        path (str): Path to the dataset file.
    Returns:
        tuple: (input dimension, output dimension, max sequence length)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            if '|' not in line:
                continue
            ipt, opt = line.split('|')
            in_dim = len(ipt.split(';')[0].split(','))
            out_dim = len(opt.split(';')[0].split(','))
            seq_len = len(ipt.split(';'))
            pred_len = len(opt.split(';'))
            max_len = max(seq_len, pred_len)
            break

    return in_dim, out_dim, seq_len, pred_len, max_len


def _normalize_data(data):
    """
    Applies feature-wise normalization to input sequences using StandardScaler.
    This function is kept for backward compatibility but should be replaced with
    global normalization in the dataset classes.

    Args:
        data (list): List of sequences where each sequence is a list of feature vectors.

    Returns:
        list: Normalized sequences.
    """
    data = np.array(data, dtype=np.float32)

    # Apply normalization only if the indices exist
    # mins = np.min(data, axis=0)
    # maxs = np.max(data, axis=0)
    # ranges = maxs - mins
    # ranges[ranges == 0] = 1.0
    # normalized_data = (data - mins) / ranges
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero
    normalized_data = (data - means) / stds

    return normalized_data.tolist()  # Convert back to list


class LazySequenceDataset(Dataset):
    """
    A lazy-loading PyTorch Dataset that reads one sample per __getitem__.
    It scans the file once to record the byte offsets for lines containing '|'.
    Optionally, a subset of indices may be provided.
    """
    def __init__(self, path, offsets=None, indices=None, normalization=True):
        self.path = path
        # Compute or use provided offsets
        if offsets is None:
            self.offsets = []
            with open(path, 'r', encoding="utf8") as f:
                offset = f.tell()
                line = f.readline()
                while line:
                    if '|' in line:
                        self.offsets.append(offset)
                    offset = f.tell()
                    line = f.readline()
        else:
            self.offsets = offsets

        # Use all indices if not provided
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.offsets)))

        self.normalization = normalization

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map dataset index to the actual offset index
        real_idx = self.indices[idx]
        offset = self.offsets[real_idx]
        with open(self.path, 'r', encoding="utf8") as f:
            f.seek(offset)
            line = f.readline()
        if '|' not in line:
            raise ValueError("Line does not contain expected delimiter '|'")
        ipt_str, opt_str = line.split('|')
        # Parse and convert input and output sequences
        ipt = [[float(val) for val in rec.split(',')] for rec in ipt_str.strip().split(';')]
        opt = [[float(val) for val in rec.split(',')] for rec in opt_str.strip().split(';')]
        if self.normalization:
            ipt = _normalize_data(ipt)
            opt = _normalize_data(opt)
        return torch.tensor(ipt, dtype=torch.float32), torch.tensor(opt, dtype=torch.float32)
    

class CSVSequenceDataset(torch.utils.data.Dataset):
    """
    Lazy PyTorch Dataset that:
     - auto-loads a CSV (first column = timestamp, last = target)
     - builds seq_len-step input (prev-target + all context cols)
       and pred_len-step output (target only)
     - does global z-normalization using StandardScaler fitted on training data
    """
    def __init__(self, csv_path,
                 seq_len=3, pred_len=3,
                 normalization=True):
        df = pd.read_csv(csv_path)
        cols = df.columns.tolist()
        # auto-detect:
        #   timestamp = cols[0], context = cols[1:-1], target = cols[-1]
        self.context_cols = cols[1:-1]
        self.target_col  = cols[-1]

        # raw arrays
        self.context = df[self.context_cols].values.astype(float)   # shape (N, C)
        self.target  = df[self.target_col].values.astype(float)     # shape (N,)
        self.N       = len(df)

        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.norm     = normalization

        if self.norm:
            # Global target scaler - will be fitted on training data
            self.y_scaler = StandardScaler()
            # Global input scaler - will be fitted on training data
            self.x_scaler = StandardScaler()

        # valid start indices: i in [1, N - (seq_len+pred_len)]
        last = self.N - (seq_len + pred_len)
        self.starts = list(range(1, last+1)) if last >= 1 else []

    def fit_scalers(self, train_indices):
        """
        Fit the scalers on training data only.
        This should be called after creating the dataset but before using it.
        
        Args:
            train_indices: List of indices corresponding to training data
        """
        if not self.norm:
            return
            
        # Gather all training input sequences
        all_x_train = []
        all_y_train = []
        
        for idx in train_indices:
            if idx >= len(self.starts):
                continue
            s = self.starts[idx]
            
            # Build input sequence
            inp = []
            for t in range(s, s + self.seq_len):
                inp.append([self.target[t-1], *self.context[t]])
            all_x_train.append(np.array(inp))
            
            # Build output sequence
            out = [[self.target[t]] for t in range(s + self.seq_len, s + self.seq_len + self.pred_len)]
            all_y_train.append(np.array(out))
        
        # Stack all training data
        if all_x_train:
            x_train = np.concatenate(all_x_train, axis=0)  # (total_time_steps, features)
            y_train = np.concatenate(all_y_train, axis=0)  # (total_time_steps, 1)
            
            # Fit scalers on training data
            self.x_scaler.fit(x_train)
            self.y_scaler.fit(y_train)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        # build input sequence
        inp = []
        for t in range(s, s + self.seq_len):
            # prev-target + all context features
            inp.append([ self.target[t-1], *self.context[t] ])
        # build output sequence
        out = [[ self.target[t] ]
               for t in range(s + self.seq_len,
                              s + self.seq_len + self.pred_len)]
        
        if self.norm:
            # Apply global normalization using fitted scalers
            inp_arr = np.array(inp, dtype=np.float32)
            out_arr = np.array(out, dtype=np.float32)
            
            inp = self.x_scaler.transform(inp_arr).tolist()
            out = self.y_scaler.transform(out_arr).tolist()

        return (
          torch.tensor(inp, dtype=torch.float32),    # [seq_len, C+1]
          torch.tensor(out, dtype=torch.float32)     # [pred_len, 1]
        )
    
    def inverse_transform(self, data):
        """
        Convert normalized targets back to original scale.
        
        Args:
            data: Tensor of shape [batch, pred_len, 1] or [pred_len, 1] with normalized values
            
        Returns:
            Tensor of same shape with values in original scale
        """
        if not self.norm:
            return data
        
        # Handle both batch and single sample cases
        if data.dim() == 3:  # [batch, pred_len, 1]
            return torch.tensor(self.y_scaler.inverse_transform(data.numpy()), dtype=torch.float32)
        elif data.dim() == 2:  # [pred_len, 1]
            return torch.tensor(self.y_scaler.inverse_transform(data.numpy()), dtype=torch.float32)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {data.dim()}D")

class FedformerSequenceDataset(CSVSequenceDataset):
    """
    Like CSVSequenceDataset, but applies a global StandardScaler fit on the entire
    train split's inputs—and then reuses that same scaler for valid/test.
    """
    def __init__(self, csv_path, seq_len=3, pred_len=3):
        # turn off the built-in normalization since we'll handle it ourselves
        super().__init__(csv_path, seq_len=seq_len, pred_len=pred_len, normalization=False)
        
        # We'll use the same global normalization approach as CSVSequenceDataset
        self.norm = True
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit_scalers(self, train_indices):
        """
        Fit the scalers on training data only.
        This should be called after creating the dataset but before using it.
        
        Args:
            train_indices: List of indices corresponding to training data
        """
        if not self.norm:
            return
            
        # Gather all training input sequences
        all_x_train = []
        all_y_train = []
        
        for idx in train_indices:
            if idx >= len(self.starts):
                continue
            s = self.starts[idx]
            
            # Build input sequence
            inp = []
            for t in range(s, s + self.seq_len):
                inp.append([self.target[t-1], *self.context[t]])
            all_x_train.append(np.array(inp))
            
            # Build output sequence
            out = [[self.target[t]] for t in range(s + self.seq_len, s + self.seq_len + self.pred_len)]
            all_y_train.append(np.array(out))
        
        # Stack all training data
        if all_x_train:
            x_train = np.concatenate(all_x_train, axis=0)  # (total_time_steps, features)
            y_train = np.concatenate(all_y_train, axis=0)  # (total_time_steps, 1)
            
            # Fit scalers on training data
            self.x_scaler.fit(x_train)
            self.y_scaler.fit(y_train)

    def __getitem__(self, idx):
        # get the raw (un-normalized) data
        X, Y = super().__getitem__(idx)         # X: [seq_len, features], Y: [pred_len, 1]

        # apply the fitted scaler to X and Y
        X_scaled = self.x_scaler.transform(X.numpy())
        Y_scaled = self.y_scaler.transform(Y.numpy())

        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(Y_scaled, dtype=torch.float32)

    def inverse_transform(self, data):
        """
        Convert normalized data back to original scale.
        
        Args:
            data: Tensor with normalized values
            
        Returns:
            Tensor with values in original scale
        """
        if hasattr(self, 'x_scaler') and hasattr(self, 'y_scaler'):
            # For input data that was scaled with StandardScaler
            if data.dim() == 3:  # [batch, seq_len, features]
                return torch.tensor(self.x_scaler.inverse_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape), dtype=torch.float32)
            elif data.dim() == 2:  # [seq_len, features]
                return torch.tensor(self.x_scaler.inverse_transform(data.numpy()), dtype=torch.float32)
        
        # For target data, use parent's inverse_transform
        return super().inverse_transform(data)


class Corpus:
    """
    Corpus class that loads the dataset and splits it into training, validation, and test sets.
    """
    def __init__(self, path, train_size=0.8, valid_size=0.1, test_size=0.1, normalization=True):
        total = train_size + valid_size + test_size
        train_size /= total
        valid_size /= total
        test_size /= total

        self.in_dim, self.out_dim, self.seq_len, self.pred_len, self.max_len = _init_dim(path)
        
        base_dataset = LazySequenceDataset(path, normalization=True)
        total_samples = len(base_dataset)
        indices = list(range(total_samples))
        random.shuffle(indices)
        train_cnt = int(total_samples * train_size)
        valid_cnt = int(total_samples * valid_size)

        train_indices = indices[:train_cnt]
        valid_indices = indices[train_cnt:train_cnt + valid_cnt]
        test_indices = indices[train_cnt + valid_cnt:]

        # Create lazy datasets sharing the precomputed offsets
        self.train = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=train_indices, normalization=normalization)
        self.valid = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=valid_indices, normalization=normalization)
        self.test  = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=test_indices, normalization=normalization)


        if len(self.train) == 0 or len(self.valid) == 0 or len(self.test) == 0:
            raise ValueError("Empty dataset split! Adjust the train/valid/test ratios.")

def default_collate_fn(batch):
    """
    Returns batch of raw (X, Y) pairs without model-specific logic.
    """
    X, Y = zip(*batch)
    X = torch.stack(X)
    Y = torch.stack(Y)
    return X, Y  # shapes: [batch, seq_len, in_dim], [batch, pred_len, out_dim]

def get_dataloaders(path,
                    batch_size=32,
                    shuffle=True,
                    train_size=0.8,
                    valid_size=0.1,
                    test_size=0.1,
                    model_type='lstm',
                    normalization=True,
                    seq_len=None,
                    pred_len=None):
    """
    Detects .csv → uses CSVSequenceDataset;
    else falls back to TXT-based Corpus.
    """
    if path.lower().endswith('.csv'):
        if seq_len is None:
            seq_len = 3
        if pred_len is None:
            pred_len = 3

        if model_type.lower() == 'fedformer':
            ds = FedformerSequenceDataset(path, seq_len=seq_len, pred_len=pred_len)
        else:
            ds = CSVSequenceDataset(path,
                                    seq_len=seq_len,
                                    pred_len=pred_len,
                                    normalization=normalization)
        
        N    = len(ds)
        idxs = list(range(N))
        random.shuffle(idxs)
        n1 = int(N * train_size)
        n2 = int(N * valid_size)

        train_indices = idxs[:n1]
        valid_indices = idxs[n1:n1+n2]
        test_indices  = idxs[n1+n2:]

        # Fit scalers on training data only (for global normalization)
        if normalization:
            ds.fit_scalers(train_indices)

        train_ds = Subset(ds, train_indices)
        valid_ds = Subset(ds, valid_indices)
        test_ds  = Subset(ds, test_indices)

        ds_train, ds_valid, ds_test = train_ds, valid_ds, test_ds
        in_dim   = 1 + len(ds.context_cols)
        out_dim  = 1
        seq_len  = ds.seq_len
        pred_len = ds.pred_len
    else:
        corpus = Corpus(path, train_size, valid_size, test_size, normalization=normalization)
        ds_train, ds_valid, ds_test = corpus.train, corpus.valid, corpus.test
        in_dim, out_dim, seq_len, pred_len, _ = _init_dim(path)

    # now build the three DataLoaders with your existing collate logic
    return (
    DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle,  num_workers=4, collate_fn=default_collate_fn),
    DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=default_collate_fn, drop_last=True),
    DataLoader(ds_test,  batch_size=1,          shuffle=False, num_workers=4, collate_fn=default_collate_fn)
)
    
