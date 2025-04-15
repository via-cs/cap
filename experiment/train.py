import torch
import torch.optim as optim
import torch.nn as nn
from CAP.cap.data.data import get_dataloaders, Corpus
from cap.models.lstm import TimeSeriesLSTM
from result import load_model, evaluate_model
from cap.models.transformer import TimeSeriesTransformer
from cap.models.Autoformer import Autoformer
from cap.models.Informer import Informer
from cap.models.FEDFormer import FEDformer
import yaml
import argparse


def train_model(train_loader, valid_loader, input_dim, output_dim, seq_len, pred_len, hidden_dim=128, num_layers=2, epochs=1, lr=0.01, patience=5, device="cuda" if torch.cuda.is_available() else "cpu", model_type='lstm'):
    """
    Trains an LSTM model for time-series forecasting.

    Args:
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden_dim (int): Number of LSTM hidden units.
        num_layers (int): Number of LSTM layers.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): 'cuda' or 'cpu'.
    """
    if model_type == 'lstm':
        model = TimeSeriesLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, d_ff=2048, num_layers=3, dropout=0.1).to(device)
    elif model_type == 'autoformer':
        model = Autoformer(input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, d_ff=2048, num_layers=3, dropout=0.1).to(device)
    elif model_type == 'informer':
        model = Informer(input_dim, input_dim, seq_len, pred_len, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=2048, factor=5, dropout=0.1, activation='gelu').to(device)
    elif model_type == 'fedformer':
        model = FEDformer(input_dim, input_dim, pred_len, output_dim, seq_len, label_len = 12, d_model=512, n_heads=8, d_ff=2048, num_layers=3, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            if model_type == 'lstm' or model_type == 'transformer':
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            elif model_type == 'autoformer' or model_type == 'informer' or model_type == 'fedformer':
                x_enc, x_dec, y = batch  # Unpack Autoformer inputs
                x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                outputs = model(x_enc, None, x_dec, None)  # Autoformer does not require x_mask
                # print(outputs)
                loss = criterion(outputs[:, -pred_len:, :], y)  # Compute loss only for forecasted values


            # inputs, targets = inputs.to(device), targets.to(device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                if model_type == 'lstm' or model_type == 'transformer':
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                elif model_type == 'autoformer' or model_type == 'informer' or model_type == 'fedformer':
                    x_enc, x_dec, y = batch
                    x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                    outputs = model(x_enc, None, x_dec, None)
                    loss = criterion(outputs[:, -pred_len:, :], y)
                
                valid_loss += loss.item()
                
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        
        # Check for early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict()  # Save the best model
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best Valid Loss: {best_valid_loss:.4f}")
            break  # Stop training
            
    # Restore the best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Best model restored.")
        
    return model
