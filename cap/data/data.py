import os
import numpy as np
import torch
from random import shuffle
from torch.utils.data import Dataset, DataLoader

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
        shuffle(indices)
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


def collate_fn_informer(batch):
    """
    Custom collate function for Informer model.
    Expects batch to contain:
    - past input features
    - past timestamps
    - future input features (zero-padded)
    - future timestamps
    - target values
    
    Args:
        batch (list of tuples): Each tuple contains:
            - x_enc (Tensor) : [seq_len, input_dim]
            - x_mark_enc (Tensor) : [seq_len, time_dim]
            - x_dec (Tensor) : [pred_len, input_dim]
            - x_mark_dec (Tensor) : [pred_len, time_dim]
            - y (Tensor) : [pred_len, 1]

    Returns:
        - x_enc (Tensor) : [batch, seq_len, input_dim]
        - x_mark_enc (Tensor) : [batch, seq_len, time_dim]
        - x_dec (Tensor) : [batch, pred_len, input_dim]
        - x_mark_dec (Tensor) : [batch, pred_len, time_dim]
        - y (Tensor) : [batch, pred_len, 1]
    """
    ipt_batch, opt_batch = zip(*batch)
    ipt_tensor = torch.stack([torch.tensor(ipt, dtype=torch.float32) for ipt in ipt_batch])
    opt_tensor = torch.stack([torch.tensor(opt, dtype=torch.float32) for opt in opt_batch])

    # Convert to batched tensors
    # x_enc = torch.stack(i)  # Shape: (batch_size, seq_len, input_dim)
    # x_mark_enc = torch.stack(x_mark_enc)  # Shape: (batch_size, seq_len, time_dim)
    # x_dec = torch.stack(x_dec)  # Shape: (batch_size, pred_len, input_dim)
    # x_mark_dec = torch.stack(x_mark_dec)  # Shape: (batch_size, pred_len, time_dim)
    # y = torch.stack(y)  # Shape: (batch_size, pred_len, 1)

    return ipt_tensor, ipt_tensor, opt_tensor


def collate_fn_autoformer(batch):
    """
    Custom collate function to reshape batch data into (seq_len, batch_size, feature_dim).
    """ 
    ipt_batch, opt_batch = zip(*batch)

    # Convert to tensor
    x_enc = torch.stack([torch.tensor(ipt, dtype=torch.float32) for ipt in ipt_batch])
    y = torch.stack([torch.tensor(opt, dtype=torch.float32) for opt in opt_batch])

    label_len = x_enc.shape[1] // 2  # Ensure label_len is valid
    # x_enc = x_enc[:, :, 0].unsqueeze(-1)
    # print(x_enc.shape)

    # **Create `x_dec`**
    x_dec = torch.cat([
        x_enc[:, -label_len:, 0].unsqueeze(-1),  # Take last `label_len` values
        torch.zeros_like(y)  # Placeholder zeros
    ], dim=1)

    # print(y)

    # print(x_enc.shape, x_dec.shape, y.shape)

    # **Fix Permute Order**
    # Autoformer expects (batch_size, seq_len, feature_dim) not (batch_size, feature_dim, seq_len)
    # Remove previous `permute` operation
    return x_enc, x_dec, y


def collate_fn(batch):
    """
    Custom collate function to reshape batch data into (seq_len, batch_size, feature_dim).
    """
    ipt_batch, opt_batch = zip(*batch)
    ipt_tensor = torch.stack([torch.tensor(ipt, dtype=torch.float32) for ipt in ipt_batch])
    opt_tensor = torch.stack([torch.tensor(opt, dtype=torch.float32) for opt in opt_batch])
    # Permute to shape: (seq_len, batch_size, feature_dim)
    ipt_tensor = ipt_tensor.permute(1, 0, 2)
    opt_tensor = opt_tensor.permute(1, 0, 2)
    return ipt_tensor, opt_tensor


def get_dataloaders(path, batch_size=32, shuffle=True, train_size=0.8, valid_size=0.1, test_size=0.1, model_type='lstm', normalization=True):
    """
    Creates DataLoaders for training, validation, and testing.
    Args:
        path (str): Path to the dataset file.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    corpus = Corpus(path, train_size, valid_size, test_size, normalization=normalization)
    if model_type in ['lstm', 'autoformer', 'informer']:
        if model_type == 'autoformer':
            train_loader = DataLoader(corpus.train, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_fn_autoformer)
            valid_loader = DataLoader(corpus.valid, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_autoformer, drop_last=True)
            test_loader = DataLoader(corpus.test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_autoformer)
        elif model_type == 'informer' or model_type == 'fedformer':
            train_loader = DataLoader(corpus.train, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_fn_informer)
            valid_loader = DataLoader(corpus.valid, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_informer, drop_last=True)        
            test_loader = DataLoader(corpus.test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_informer)
        else: 
            train_loader = DataLoader(corpus.train, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_fn)
            valid_loader = DataLoader(corpus.valid, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, drop_last=True)
            test_loader = DataLoader(corpus.test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(corpus.train, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        valid_loader = DataLoader(corpus.valid, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        test_loader = DataLoader(corpus.test, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader
