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
            max_len = max(len(ipt.split(';')), len(opt.split(';')))
            break

    return in_dim, out_dim, max_len


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
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized_data = (data - mins) / ranges

    return normalized_data.tolist()  # Convert back to list


# class SequenceDataset(Dataset):
#     """
#     A PyTorch Dataset class that returns input-output pairs for each sequence.
#     """
#     def __init__(self, sequences):
#         """
#         Initializes the dataset with input-output pairs.
#         Args:
#             sequences (list): List of [input, output] pairs.
#         """
#         self.sequences = sequences

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         """
#         Returns the input-output pair at the given index.
#         Args:
#             idx (int): Index of the sequence.
#         Returns:
#             torch.Tensor: Input and output tensors.
#         """
#         ipt, opt = self.sequences[idx]
#         return torch.tensor(ipt, dtype=torch.float16), torch.tensor(opt, dtype=torch.float16)

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
        return ipt, opt

class Corpus:
    """
    Corpus class that loads the dataset and splits it into training, validation, and test sets.
    """
    def __init__(self, path, train_size=0.8, valid_size=0.1, test_size=0.1, normalization=True):
        total = train_size + valid_size + test_size
        train_size /= total
        valid_size /= total
        test_size /= total

        self.in_dim, self.out_dim, self.max_len = _init_dim(path)
        # dataset = self.load(path)
        # shuffle(dataset)
        base_dataset = LazySequenceDataset(path, normalization=True)
        total_samples = len(base_dataset)
        indices = list(range(total_samples))
        shuffle(indices)

        # train_cnt = int(len(dataset) * train_size)
        # valid_cnt = int(len(dataset) * valid_size)
        train_cnt = int(total_samples * train_size)
        valid_cnt = int(total_samples * valid_size)

        train_indices = indices[:train_cnt]
        valid_indices = indices[train_cnt:train_cnt + valid_cnt]
        test_indices = indices[train_cnt + valid_cnt:]

        # Create SequenceDataset for train, validation, and test
        # self.train = SequenceDataset(dataset[:train_cnt])
        # self.valid = SequenceDataset(dataset[train_cnt:train_cnt + valid_cnt])
        # self.test = SequenceDataset(dataset[train_cnt + valid_cnt:])
        # Create lazy datasets sharing the precomputed offsets
        self.train = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=train_indices)
        self.valid = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=valid_indices)
        self.test  = LazySequenceDataset(path, offsets=base_dataset.offsets, indices=test_indices)
        self.normalization = normalization


        if len(self.train) == 0 or len(self.valid) == 0 or len(self.test) == 0:
            raise ValueError("Empty dataset split! Adjust the train/valid/test ratios.")

    # def load(self, path):
    #     """
    #     Loads the dataset from a file and returns a list of sequences.
    #     Args:
    #         path (str): Path to the dataset file.
    #     Returns:
    #         list: List of input-output sequences.
    #     """
    #     if not os.path.exists(path):
    #         raise FileNotFoundError(f"Dataset not found at {path}")

    #     seqs = []
    #     with open(path, 'r', encoding="utf8") as f:
    #         for line in f:
    #             if '|' not in line:
    #                 continue
    #             ipt, opt = line.split('|')
    #             ipt = [[float(val) for val in rec.split(',')] for rec in ipt.split(';')]
    #             opt = [[float(val) for val in rec.split(',')] for rec in opt.split(';')]

    #             # Normalize input
    #             # ipt = _normalize_data(ipt)

    #             seqs.append([ipt, opt])
    #     return seqs


def collate_fn(batch):
    """
    Custom collate function to reshape batch data into (seq_len, batch_size, feature_dim).
    """
    ipt_batch, opt_batch = zip(*batch)  # Unzip the batch into two lists

    # ipt_tensor = torch.stack(ipt_batch)  # Shape: (batch_size, seq_len, feature_dim)
    # opt_tensor = torch.stack(opt_batch)  # Shape: (batch_size, seq_len, feature_dim)
    ipt_tensor = torch.stack([torch.tensor(ipt, dtype=torch.float32) for ipt in ipt_batch])
    opt_tensor = torch.stack([torch.tensor(opt, dtype=torch.float32) for opt in opt_batch])

    # Convert to (seq_len, batch_size, feature_dim)
    ipt_tensor = ipt_tensor.permute(1, 0, 2)  # (batch_size, seq_len, feature_dim) → (seq_len, batch_size, feature_dim)
    opt_tensor = opt_tensor.permute(1, 0, 2)  # (batch_size, seq_len, feature_dim) → (seq_len, batch_size, feature_dim)

    return ipt_tensor, opt_tensor


def get_dataloaders(path, batch_size=32, shuffle=True, train_size=0.8, valid_size=0.1, test_size=0.1, normalization = True):
    """
    Creates DataLoaders for training, validation, and testing.
    Args:
        path (str): Path to the dataset file.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    corpus = Corpus(path, train_size, valid_size, test_size, normalization)

    train_loader = DataLoader(corpus.train, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(corpus.valid, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(corpus.test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader
