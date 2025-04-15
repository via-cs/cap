import pytest
import torch
from cap.models.Informer import Informer

def test_informer_initialization():
    model = Informer()
    assert isinstance(model, Informer)

def test_informer_forward():
    model = Informer()
    batch_size = 32
    seq_length = 100
    input_dim = 1
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    output = model(x)
    assert output.shape[0] == batch_size  # Check batch size
    assert output.shape[1] == seq_length  # Check sequence length 