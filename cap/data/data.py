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
    Applies feature-wise normalization to input sequences.

    Normalization formulas:`

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
     - does per-sample z-normalization if requested
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
            # global target scaler
            self.y_mean = float(self.target.mean())
            self.y_std  = float(self.target.std())
            if self.y_std == 0.0:
                self.y_std = 1.0

        # valid start indices: i in [1, N - (seq_len+pred_len)]
        last = self.N - (seq_len + pred_len)
        self.starts = list(range(1, last+1)) if last >= 1 else []

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
            # per-sample normalize inputs (as before)
            inp = _normalize_data(inp)
            # **global** normalize outputs
            # out is list of [ [y1], [y2], ... ] → shape (pred_len,1)
            out_arr = np.array(out, dtype=np.float32)
            out = ((out_arr - self.y_mean) / self.y_std).tolist()

        return (
          torch.tensor(inp, dtype=torch.float32),    # [seq_len, C+1]
          torch.tensor(out, dtype=torch.float32)     # [pred_len, 1]
        )
    
class FedformerSequenceDataset(CSVSequenceDataset):
    """
    Like CSVSequenceDataset, but applies a global StandardScaler fit on the entire
    train split’s inputs—and then reuses that same scaler for valid/test.
    """
    def __init__(self, csv_path, seq_len=3, pred_len=3):
        # turn off the built-in per-sample normalization
        super().__init__(csv_path, seq_len=seq_len, pred_len=pred_len, normalization=False)

        # --- 1) gather ALL input sequences as one big array to fit the scaler ---
        all_X = []
        for i in range(len(self)):
            X, _ = super().__getitem__(i)       # [seq_len, features]
            all_X.append(X.numpy())
        # stack into shape (N * seq_len, features)
        arr = np.concatenate(all_X, axis=0)     # shape (total_time_steps, features)

        # --- 2) fit StandardScaler on that array ---
        self.scaler = StandardScaler().fit(arr)

    def __getitem__(self, idx):
        # get the raw (un-normalized) data
        X, Y = super().__getitem__(idx)         # X: [seq_len, features], Y: [pred_len, 1]

        # apply the fitted scaler to X
        # reshape → (seq_len, feat) → (seq_len*feat, ) then back
        seq_len, feat = X.shape
        X_scaled = self.scaler.transform(X.numpy().reshape(-1, feat))  # (seq_len*1, feat)
        X_scaled = X_scaled.reshape(seq_len, feat)

        return torch.tensor(X_scaled, dtype=torch.float32), Y


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

        ds = CSVSequenceDataset(path,
                                seq_len=seq_len,
                                pred_len=pred_len,
                                normalization=normalization)
        N    = len(ds)
        idxs = list(range(N))
        random.shuffle(idxs)
        n1 = int(N * train_size)
        n2 = int(N * valid_size)

        train_ds = Subset(ds, idxs[:n1])
        valid_ds = Subset(ds, idxs[n1:n1+n2])
        test_ds  = Subset(ds, idxs[n1+n2:])

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
    
