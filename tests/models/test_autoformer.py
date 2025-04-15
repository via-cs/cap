import pytest
import torch
from cap.models.Autoformer import Autoformer

def test_autoformer_initialization():
    model = Autoformer()
    assert isinstance(model, Autoformer)

def test_autoformer_forward():
    model = Autoformer()
    batch_size = 32
    seq_length = 100
    input_dim = 1
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    output = model(x)
    assert output.shape[0] == batch_size  # Check batch size
    assert output.shape[1] == seq_length  # Check sequence length 