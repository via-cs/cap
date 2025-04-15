"""
Evaluation utilities for time series models.
"""

import torch
import torch.nn as nn
from ..models.lstm import TimeSeriesLSTM
from ..models.transformer import Transformer
from ..models.Autoformer import Autoformer
from ..models.Informer import Informer
from ..models.FEDFormer import FEDformer


def load_model(model_path, input_dim, output_dim, seq_len, pred_len, 
               hidden_dim=128, num_layers=2, 
               device="cuda" if torch.cuda.is_available() else "cpu", 
               model_type="lstm"):
    """
    Loads a trained time series forecasting model.

    Args:
        model_path (str): Path to the trained model
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
        seq_len (int): Input sequence length
        pred_len (int): Prediction sequence length
        hidden_dim (int): Number of hidden units (for LSTM)
        num_layers (int): Number of layers
        device (str): Device to load model on ('cuda' or 'cpu')
        model_type (str): Type of model to load ('lstm', 'transformer', 'autoformer', 'informer', 'fedformer')

    Returns:
        nn.Module: The loaded model
    """
    # Initialize model based on type
    if model_type == 'lstm':
        model = TimeSeriesLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    elif model_type == 'transformer':
        model = Transformer(input_dim, output_dim, seq_len, pred_len, 
                          d_model=512, n_heads=8, d_ff=2048, num_layers=3, 
                          dropout=0.1).to(device)
    elif model_type == 'autoformer':
        model = Autoformer(input_dim, output_dim, seq_len, pred_len, 
                         d_model=512, n_heads=8, d_ff=2048, num_layers=3, 
                         dropout=0.1).to(device)
    elif model_type == 'informer':
        model = Informer(input_dim, input_dim, seq_len, pred_len, 
                        d_model=512, n_heads=8, e_layers=3, d_layers=2, 
                        d_ff=2048, factor=5, dropout=0.1, activation='gelu').to(device)
    elif model_type == 'fedformer':
        model = FEDformer(input_dim, input_dim, pred_len, output_dim, seq_len, 
                         label_len=12, d_model=512, n_heads=8, d_ff=2048, 
                         num_layers=3, dropout=0.1).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu", 
                  model_type='lstm'):
    """
    Evaluates a time series forecasting model on test data.

    Args:
        model (nn.Module): The trained model
        test_loader: DataLoader for test dataset
        device (str): Device to evaluate on ('cuda' or 'cpu')
        model_type (str): Type of model being evaluated

    Returns:
        float: Mean Squared Error (MSE) loss
    """
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            if model_type in ['lstm', 'transformer']:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            else:  # autoformer, informer, fedformer
                x_enc, x_dec, y = batch
                x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                outputs = model(x_enc, None, x_dec, None)
                loss = criterion(outputs, y)
            
            total_loss += loss.item()
            num_batches += 1

    return total_loss / len(test_loader) if num_batches > 0 else float('inf') 