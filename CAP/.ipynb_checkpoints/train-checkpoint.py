import torch
import torch.optim as optim
import torch.nn as nn
from data import get_dataloaders, Corpus
from models.lstm import TimeSeriesLSTM
from result import load_model, evaluate_model
import yaml
import argparse


def train_model(train_loader, valid_loader, input_dim, output_dim, hidden_dim=128, num_layers=2, epochs=1, lr=0.01, patience=5, device="cuda" if torch.cuda.is_available() else "cpu"):
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
    model = TimeSeriesLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss / len(train_loader):.4f} | Valid Loss: {valid_loss / len(valid_loader):.4f}")
        
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
