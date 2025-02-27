import torch
import torch.nn as nn
from data import get_dataloaders
from models.lstm import TimeSeriesLSTM

def load_model(model_path, input_dim, output_dim, hidden_dim=128, num_layers=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads the trained LSTM model.

    Args:
        model_path (str): Path to the trained model.
        input_dim (int): Input feature dimension.
        output_dim (int): Output feature dimension.
        hidden_dim (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        device (str): 'cuda' or 'cpu'.

    Returns:
        nn.Module: The loaded model.
    """
    model = TimeSeriesLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluates the model on the test dataset and computes the Mean Squared Error (MSE) loss.

    Args:
        model (nn.Module): The trained model.
        test_loader: DataLoader for test dataset.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Mean Squared Error (MSE) loss.
    """
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)  # Forward pass

            loss = criterion(outputs, targets)  # Compute MSE loss
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')

