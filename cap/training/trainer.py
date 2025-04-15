"""
Training utilities for time series models.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from ..models.lstm import TimeSeriesLSTM
from ..models.transformer import Transformer
from ..models.Autoformer import Autoformer
from ..models.Informer import Informer
from ..models.FEDFormer import FEDformer


def train_model(train_loader, valid_loader, input_dim, output_dim, seq_len, pred_len, 
                hidden_dim=128, num_layers=2, epochs=1, lr=0.01, patience=5, 
                device="cuda" if torch.cuda.is_available() else "cpu", model_type='lstm'):
    """
    Trains a time series forecasting model.

    Args:
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        seq_len (int): Input sequence length
        pred_len (int): Prediction sequence length
        hidden_dim (int): Number of hidden units (for LSTM)
        num_layers (int): Number of layers
        epochs (int): Number of training epochs
        lr (float): Learning rate
        patience (int): Early stopping patience
        device (str): Device to train on ('cuda' or 'cpu')
        model_type (str): Type of model to train ('lstm', 'transformer', 'autoformer', 'informer', 'fedformer')

    Returns:
        nn.Module: The trained model
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            
            if model_type in ['lstm', 'transformer']:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            else:  # autoformer, informer, fedformer
                x_enc, x_dec, y = batch
                x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                # Pass None for time features (x_mark_enc and x_mark_dec)
                outputs = model(x_enc, None, x_dec, None)
                loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
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
                
                valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        
        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best Valid Loss: {best_valid_loss:.4f}")
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Best model restored.")
        
    return model 