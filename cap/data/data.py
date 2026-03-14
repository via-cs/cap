import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler

output_type = "multi"

def calculate_borders(total_length, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seq_len=96):
    """
    Generalized border calculation for time series data splitting.
    
    Args:
        total_length (int): Total length of the time series
        train_ratio (float): Ratio for training data (default: 0.7)
        val_ratio (float): Ratio for validation data (default: 0.15) 
        test_ratio (float): Ratio for test data (default: 0.15)
        seq_len (int): Sequence length for prediction (default: 96)
        
    Returns:
        tuple: (border1s, border2s) where each is a list of 3 integers
               [train_start, val_start, test_start] and [train_end, val_end, test_end]
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Calculate split points
    train_end = int(total_length * train_ratio)
    val_end = int(total_length * (train_ratio + val_ratio))
    test_end = total_length
    
    # Define borders for each split
    border1s = [0, train_end - seq_len, val_end - seq_len]
    border2s = [train_end, val_end, test_end]
    
    return border1s, border2s


# def _init_dim(path):
#     """
#     Initializes input/output dimensions and max sequence length based on the dataset.
#     Args:
#         path (str): Path to the dataset file.
#     Returns:
#         tuple: (input dimension, output dimension, max sequence length)
#     """
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset not found at {path}")
# 
#     with open(path, 'r', encoding="utf8") as f:
#         for line in f:
#             if '|' not in line:
#                 continue
#             ipt, opt = line.split('|')
#             in_dim = len(ipt.split(';')[0].split(','))
#             out_dim = len(opt.split(';')[0].split(','))
#             seq_len = len(ipt.split(';'))
#             pred_len = len(opt.split(';'))
#             max_len = max(seq_len, pred_len)
#             break
# 
#     return in_dim, out_dim, seq_len, pred_len, max_len


# def _normalize_data(data):
#     """
#     Applies feature-wise normalization to input sequences using StandardScaler.
#     This function is kept for backward compatibility but should be replaced with
#     global normalization in the dataset classes.
# 
#     Args:
#         data (list): List of sequences where each sequence is a list of feature vectors.
# 
#     Returns:
#         list: Normalized sequences.
#     """
#     data = np.array(data, dtype=np.float32)
# 
#     # Apply normalization only if the indices exist
#     # mins = np.min(data, axis=0)
#     # maxs = np.max(data, axis=0)
#     # ranges = maxs - mins
#     # ranges[ranges == 0] = 1.0
#     # normalized_data = (data - mins) / ranges
#     means = np.mean(data, axis=0)
#     stds = np.std(data, axis=0)
#     stds[stds == 0] = 1.0  # Avoid division by zero
#     normalized_data = (data - means) / stds
# 
#     return normalized_data.tolist()  # Convert back to list


# class LazySequenceDataset(Dataset):
#     """
#     A lazy-loading PyTorch Dataset that reads one sample per __getitem__.
#     It scans the file once to record the byte offsets for lines containing '|'.
#     Optionally, a subset of indices may be provided.
#     """
#     def __init__(self, path, offsets=None, indices=None, normalization=True):
#         self.path = path
#         # Compute or use provided offsets
#         if offsets is None:
#             self.offsets = []
#             with open(path, 'r', encoding="utf8") as f:
#                 offset = f.tell()
#                 line = f.readline()
#                 while line:
#                     if '|' in line:
#                         self.offsets.append(offset)
#                     offset = f.tell()
#                     line = f.readline()
#         else:
#             self.offsets = offsets
# 
#         # Use all indices if not provided
#         if indices is not None:
#             self.indices = indices
#         else:
#             self.indices = list(range(len(self.offsets)))
# 
#         self.normalization = normalization
# 
#     def __len__(self):
#         return len(self.indices)
# 
#     def __getitem__(self, idx):
#         # Map dataset index to the actual offset index
#         real_idx = self.indices[idx]
#         offset = self.offsets[real_idx]
#         with open(self.path, 'r', encoding="utf8") as f:
#             f.seek(offset)
#             line = f.readline()
#         if '|' not in line:
#             raise ValueError("Line does not contain expected delimiter '|'")
#         ipt_str, opt_str = line.split('|')
#         # Parse and convert input and output sequences
#         ipt = [[float(val) for val in rec.split(',')] for rec in ipt_str.strip().split(';')]
#         opt = [[float(val) for val in rec.split(',')] for rec in opt_str.strip().split(';')]
#         if self.normalization:
#             ipt = _normalize_data(ipt)
#             opt = _normalize_data(opt)
#         return torch.tensor(ipt, dtype=torch.float32), torch.tensor(opt, dtype=torch.float32)
    

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
                 normalization=True, train_size=0.8, valid_size=0.1, test_size=0.1):
        df = pd.read_csv(csv_path)
        cols = df.columns.tolist()
        # auto-detect:
        #   timestamp = cols[0], context = cols[1:-1], target = cols[-1]
        self.context_cols = cols[1:-1]
        self.target_col  = cols[-1]

        # raw arrays
        self.context = df[self.context_cols].values.astype(float)   # shape (N, C)
        self.target  = df[self.target_col].values.astype(float)     # shape (N,)
        self.input_data = np.concatenate([self.target.reshape(-1, 1), self.context], axis=1)

        self.border1s, self.border2s = calculate_borders(len(df), train_size, valid_size, test_size, seq_len=seq_len)

        self.N       = len(df)
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.norm     = normalization

        if self.norm:
            self.scaler = StandardScaler()
            self.scaler.fit(self.input_data[self.border1s[0]:self.border2s[0]])
            self.input_data = self.scaler.transform(self.input_data)
            self.target = self.input_data[:, 0]
            self.context = self.input_data[:, 1:]

        # valid start indices: i in [1, N - (seq_len+pred_len)]
        last = self.N - (seq_len + pred_len)
        self.starts = list(range(1, last+1)) if last >= 1 else []



    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        
        # Build input sequence
        inp = []
        for t in range(s, s + self.seq_len):
            # target as first feature + all context features
            inp.append([ self.target[t-1], *self.context[t-1] ])
        # build output sequence
        if output_type == 'multi':
            out = []
            for t in range(s + self.seq_len, s + self.seq_len + self.pred_len):
                out.append([self.target[t], *self.context[t]])
        else:
            out = [[ self.target[t] ]
                for t in range(s + self.seq_len,
                                s + self.seq_len + self.pred_len)]

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
            # Extract target column (first column) from the data
            target_data = data[:, :, 0:1]  # Keep the same shape
            return torch.tensor(self.scaler.inverse_transform(target_data.reshape(-1, 1)).reshape(data.shape), dtype=torch.float32)
        elif data.dim() == 2:  # [pred_len, 1]
            return torch.tensor(self.scaler.inverse_transform(data.numpy()), dtype=torch.float32)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {data.dim()}D")


# class Corpus:
#     """
#     Corpus class that loads the dataset and splits it into training, validation, and test sets.
#     """
#     def __init__(self, path, train_size=0.8, valid_size=0.1, test_size=0.1, normalization=True):
#         total = train_size + valid_size + test_size
#         train_size /= total
#         valid_size /= total
#         test_size /= total
# 
#         self.in_dim, self.out_dim, self.seq_len, self.pred_len, self.max_len = _init_dim(path)
#         
#         base_dataset = LazySequenceDataset(path, normalization=True)
#         total_samples = len(base_dataset)
#         indices = list(range(total_samples))
#         random.shuffle(indices)
#         train_cnt = int(total_samples * train_size)
#         valid_cnt = int(total_samples * valid_size)
# 
#         train_indices = indices[:train_cnt]
#         valid_indices = indices[train_cnt:train_cnt + valid_cnt]
#         test_indices = indices[train_cnt + valid_cnt:]
# 
#         # Create lazy datasets sharing the precomputed offsets
#         self.train = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=train_indices, normalization=normalization)
#         self.valid = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=valid_indices, normalization=normalization)
#         self.test  = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=test_indices, normalization=normalization)
# 
# 
#         if len(self.train) == 0 or len(self.valid) == 0 or len(self.test) == 0:
#             raise ValueError("Empty dataset split! Adjust the train/valid/test ratios.")

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
    Creates dataloaders for CSV files using CSVSequenceDataset.
    """
    # Only support CSV files now
    if not path.lower().endswith('.csv'):
        raise ValueError("Only CSV files are supported. Please convert your data to CSV format.")

    ds = CSVSequenceDataset(path,
                            seq_len=seq_len,
                            pred_len=pred_len,
                            normalization=normalization, train_size=train_size, valid_size=valid_size, test_size=test_size)
    
    # Use border-based splitting (chronological order)
    # Map border indices to indices in the starts list
    train_indices = [i for i, start in enumerate(ds.starts) if ds.border1s[0] <= start <= ds.border2s[0]]
    valid_indices = [i for i, start in enumerate(ds.starts) if ds.border1s[1] <= start <= ds.border2s[1]]
    test_indices  = [i for i, start in enumerate(ds.starts) if ds.border1s[2] <= start <= ds.border2s[2] - seq_len]

    train_ds = Subset(ds, train_indices)
    valid_ds = Subset(ds, valid_indices)
    test_ds  = Subset(ds, test_indices)

    #     ds_train, ds_valid, ds_test = train_ds, valid_ds, test_ds
    #     in_dim   = 1 + len(ds.context_cols)
    #     out_dim  = 1
    #     seq_len  = ds.seq_len
    #     pred_len = ds.pred_len
    # else:
    #     corpus = Corpus(path, train_size, valid_size, test_size, normalization=normalization)
    #     ds_train, ds_valid, ds_test = corpus.train, corpus.valid, corpus.test
    #     in_dim, out_dim, seq_len, pred_len, _ = _init_dim(path)

    # now build the three DataLoaders with your existing collate logic
    return (
    DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,  num_workers=4, collate_fn=default_collate_fn),
    DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=default_collate_fn, drop_last=True),
    DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=4, collate_fn=default_collate_fn)
)
    
