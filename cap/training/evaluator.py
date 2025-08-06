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
from ..models.iTransformer import iTransformer


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
        model_type (str): Type of model to load ('lstm', 'transformer', 'autoformer', 'informer', 'fedformer', 'itransformer')

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
    elif model_type == 'itransformer':
        model = iTransformer(input_dim, output_dim, seq_len, pred_len, 
                           d_model=512, n_heads=8, d_ff=2048, num_layers=3, 
                           dropout=0.05, embed="fixed", freq="h", factor=3, 
                           activation="gelu").to(device)
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


def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu", model_type="lstm"):
    """
    Evaluate a trained model on test_loader. Returns average MSE on both normalized and original scale.
    Works for LSTM, Transformer, Autoformer, Informer, FEDformer, TimesNet, iTransformer, etc.
    """
    model_type = model_type.lower()
    model.to(device).eval()
    criterion = nn.MSELoss(reduction='mean')
    total_loss_norm = 0.0
    total_loss_orig = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            # move all tensors in the batch to device
            batch = tuple(t.to(device) for t in batch)

            # use the model's .prepare_batch to unpack and handle model-specific logic
            inputs, target = model.prepare_batch(batch)
            output = model(*inputs)

            # LSTM returns all timesteps → keep only the final pred_len steps
            if model_type == 'lstm':
                pred_len = target.shape[1]
                output = output[:, -pred_len:, :]
            
            # FEDformer & TimesNet return one channel per input feature → keep only the target (first) channel
            if model_type in ('fedformer', 'timesnet', 'itransformer'):
                # slice off extra feature channels, keep only the 1-d target
                output = output[..., :1]

            # compute MSE on normalized data
            batch_size = target.size(0)
            loss_norm = criterion(output, target) * batch_size
            total_loss_norm += loss_norm.item()
            
            # compute MSE on original scale data
            try:
                # Get the dataset from the dataloader
                dataset = test_loader.dataset
                
                # Handle Subset wrapper (from torch.utils.data.Subset)
                if hasattr(dataset, 'dataset'):
                    # If dataset is wrapped in Subset, get the underlying dataset
                    dataset = dataset.dataset
                
                if hasattr(dataset, 'inverse_transform'):
                    # Convert predictions and targets to original scale
                    output_orig = dataset.inverse_transform(output.cpu())
                    target_orig = dataset.inverse_transform(target.cpu())
                    
                    # Compute MSE on original scale
                    loss_orig = criterion(output_orig, target_orig) * batch_size
                    total_loss_orig += loss_orig.item()
                else:
                    # If no inverse_transform available, use normalized MSE
                    total_loss_orig += loss_norm.item()
            except Exception as e:
                # If inverse_transform fails, use normalized MSE
                print(f"Warning: Could not compute original scale MSE: {e}")
                total_loss_orig += loss_norm.item()
            
            n_samples += batch_size

    avg_mse_norm = total_loss_norm / n_samples
    avg_mse_orig = total_loss_orig / n_samples
    
    print(f"Test MSE (normalized): {avg_mse_norm:.6f}")
    print(f"Test MSE (original scale): {avg_mse_orig:.6f}")
    
    return avg_mse_orig  # Return original scale MSE for fair comparison
