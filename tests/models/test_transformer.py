import pytest
import torch
from cap.models import Transformer

def test_transformer_init():
    """Test if transformer model can be initialized correctly."""
    model = Transformer(
        input_dim=10,
        output_dim=1,
        seq_len=96,
        pred_len=24
    )
    assert isinstance(model, Transformer)

def test_transformer_forward():
    """Test if transformer model can perform forward pass."""
    batch_size = 32
    seq_len = 96
    input_dim = 10
    pred_len = 24
    output_dim = 1

    model = Transformer(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    assert output.shape == (batch_size, pred_len, output_dim)

    assert output.shape[0] == batch_size  # Check batch size
    assert output.shape[1] == pred_len  # Check prediction length 