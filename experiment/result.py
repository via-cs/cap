import torch
import torch.nn as nn
from CAP.cap.data.data import get_dataloaders
from cap.models.lstm import TimeSeriesLSTM
from cap.models.transformer import TimeSeriesTransformer
from cap.models.Autoformer import Autoformer
from cap.models.Informer import Informer
from cap.models.FEDFormer import FEDformer

def load_model(model_path, input_dim, output_dim, seq_len,pred_len,hidden_dim=128, num_layers=2, device="cuda" if torch.cuda.is_available() else "cpu", model_type="lstm"):
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
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu", model_type='lstm'):
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
        # for inputs, targets in test_loader:
        #     inputs, targets = inputs.to(device), targets.to(device)

        #     outputs = model(inputs)  # Forward pass

        #     loss = criterion(outputs, targets)  # Compute MSE loss
        #     total_loss += loss.item()
        #     num_batches += 1
        for batch in test_loader:
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
                loss = criterion(outputs[:, -y.shape[1]:, :], y)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / len(test_loader) if num_batches > 0 else float('inf')

