import pytest
import torch
from cap.models.FEDFormer import FEDFormer

def test_fedformer_initialization():
    model = FEDFormer()
    assert isinstance(model, FEDFormer)

def test_fedformer_forward():
    model = FEDFormer()
    batch_size = 32
    seq_length = 100
    input_dim = 1
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    output = model(x)
    assert output.shape[0] == batch_size  # Check batch size
    assert output.shape[1] == seq_length  # Check sequence length 