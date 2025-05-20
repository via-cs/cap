import torch.nn as nn

class BaseTimeSeriesModel(nn.Module):
    def prepare_batch(self, batch):
        """Unpack and optionally move batch to model's device"""
        X, Y = batch
        return X, Y
