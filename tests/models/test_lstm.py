import pytest
import torch
from cap.models.lstm import TimeSeriesLSTM

def test_lstm_initialization():
    """Test if LSTM model can be initialized correctly."""
    model = TimeSeriesLSTM()  # Using default parameters
    assert isinstance(model, TimeSeriesLSTM)
    assert model.hidden_dim == 64  # Check default hidden dimension
    assert model.num_layers == 2  # Check default number of layers

def test_lstm_forward():
    """Test if LSTM model can perform forward pass."""
    batch_size = 32
    seq_length = 100
    input_dim = 3
    hidden_dim = 10
    output_dim = 1
    
    model = TimeSeriesLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # Create input tensor
    x = torch.randn(seq_length, batch_size, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (seq_length, batch_size, output_dim)
    assert not torch.isnan(output).any()  # Check for NaN values
    assert not torch.isinf(output).any()  # Check for Inf values 