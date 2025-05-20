"""
Tests for the training module.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cap.training import train_model, evaluate_model, load_model


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    # Create dummy time series data
    batch_size = 32
    seq_len = 24
    pred_len = 12
    input_dim = 10
    output_dim = 1
    
    # Generate random data
    X = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, pred_len, output_dim)
    
    # Create datasets
    train_dataset = TensorDataset(X, y)
    valid_dataset = TensorDataset(X, y)
    test_dataset = TensorDataset(X, y)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'seq_len': seq_len,
        'pred_len': pred_len
    }


@pytest.fixture
def dummy_autoformer_data():
    """Create dummy data for Autoformer/Informer/FEDformer testing."""
    batch_size = 32
    seq_len = 24
    pred_len = 12
    input_dim = 10
    output_dim = 1
    
    # Generate random data
    x_enc = torch.randn(batch_size, seq_len, input_dim)  # Encoder input
    x_mark_enc = None  # Time features for encoder (optional)
    x_dec = torch.randn(batch_size, pred_len, input_dim)  # Decoder input
    x_mark_dec = None  # Time features for decoder (optional)
    y = torch.randn(batch_size, pred_len, output_dim)  # Target values
    
    # Create datasets with None for optional time features
    train_dataset = TensorDataset(x_enc, x_dec, y)
    valid_dataset = TensorDataset(x_enc, x_dec, y)
    test_dataset = TensorDataset(x_enc, x_dec, y)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'seq_len': seq_len,
        'pred_len': pred_len
    }


def test_train_lstm(dummy_data):
    """Test training LSTM model."""
    model = train_model(
        train_loader=dummy_data['train_loader'],
        valid_loader=dummy_data['valid_loader'],
        input_dim=dummy_data['input_dim'],
        output_dim=dummy_data['output_dim'],
        seq_len=dummy_data['seq_len'],
        pred_len=dummy_data['pred_len'],
        hidden_dim=64,
        num_layers=2,
        epochs=2,
        lr=0.001,
        patience=3,
        model_type='lstm'
    )
    
    assert isinstance(model, nn.Module)
    assert hasattr(model, 'forward')


def test_train_transformer(dummy_data):
    """Test training Transformer model."""
    model = train_model(
        train_loader=dummy_data['train_loader'],
        valid_loader=dummy_data['valid_loader'],
        input_dim=dummy_data['input_dim'],
        output_dim=dummy_data['output_dim'],
        seq_len=dummy_data['seq_len'],
        pred_len=dummy_data['pred_len'],
        epochs=2,
        lr=0.001,
        patience=3,
        model_type='transformer'
    )
    
    assert isinstance(model, nn.Module)
    assert hasattr(model, 'forward')


def test_train_autoformer(dummy_autoformer_data):
    """Test training Autoformer model."""
    model = train_model(
        train_loader=dummy_autoformer_data['train_loader'],
        valid_loader=dummy_autoformer_data['valid_loader'],
        input_dim=dummy_autoformer_data['input_dim'],
        output_dim=dummy_autoformer_data['output_dim'],
        seq_len=dummy_autoformer_data['seq_len'],
        pred_len=dummy_autoformer_data['pred_len'],
        epochs=2,
        lr=0.001,
        patience=3,
        model_type='autoformer'
    )
    
    assert isinstance(model, nn.Module)
    assert hasattr(model, 'forward')


def test_evaluate_model(dummy_data):
    """Test model evaluation."""
    # Train a model first
    model = train_model(
        train_loader=dummy_data['train_loader'],
        valid_loader=dummy_data['valid_loader'],
        input_dim=dummy_data['input_dim'],
        output_dim=dummy_data['output_dim'],
        seq_len=dummy_data['seq_len'],
        pred_len=dummy_data['pred_len'],
        epochs=2,
        lr=0.001,
        patience=3,
        model_type='lstm'
    )
    
    # Evaluate the model
    test_loss = evaluate_model(
        model=model,
        test_loader=dummy_data['test_loader'],
        model_type='lstm'
    )
    
    assert isinstance(test_loss, float)
    assert test_loss >= 0


def test_load_model(dummy_data, tmp_path):
    """Test model loading and saving."""
    # Train a model first
    model = train_model(
        train_loader=dummy_data['train_loader'],
        valid_loader=dummy_data['valid_loader'],
        input_dim=dummy_data['input_dim'],
        output_dim=dummy_data['output_dim'],
        seq_len=dummy_data['seq_len'],
        pred_len=dummy_data['pred_len'],
        epochs=2,
        lr=0.001,
        patience=3,
        model_type='lstm'
    )
    
    # Save the model
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Load the model
    loaded_model = load_model(
        model_path=str(model_path),
        input_dim=dummy_data['input_dim'],
        output_dim=dummy_data['output_dim'],
        seq_len=dummy_data['seq_len'],
        pred_len=dummy_data['pred_len'],
        model_type='lstm'
    )
    
    assert isinstance(loaded_model, nn.Module)
    assert hasattr(loaded_model, 'forward')
    
    # Test that loaded model produces same outputs
    for batch in dummy_data['test_loader']:
        inputs, _ = batch
        with torch.no_grad():
            original_output = model(inputs)
            loaded_output = loaded_model(inputs)
            assert torch.allclose(original_output, loaded_output) 