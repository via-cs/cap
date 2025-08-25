"""
Training utilities for time series models:
train_loader (DataLoader): training dataset loader yielding batches.
valid_loader (DataLoader): validation dataset loader.
input_dim (int): number of features in the input sequences.
output_dim (int): number of features in the output (forecast) sequences.
seq_len (int): length of input sequence window.
pred_len (int): length of prediction horizon.
hidden_dim (int): hidden dimension (used for LSTM or other models that require it).
num_layers (int): number of layers in the model (if applicable).
epochs (int): training epochs.
lr (float): learning rate for the optimizer.
patience (int): early stopping patience.
device (str): device to train on ("cpu" or "cuda").
model_type (str): identifier of which model to train (e.g. "lstm", "transformer", etc.).
"""

import inspect
import torch
import torch.optim as optim
import torch.nn as nn
from ..models.lstm       import TimeSeriesLSTM
from ..models.transformer import Transformer
from ..models.Autoformer  import Autoformer
from ..models.Informer    import Informer
from ..models.FEDFormer   import FEDformer
from ..models.TimesNet    import TimesNet
from ..models.iTransformer import iTransformer

def train_model(
    train_loader, valid_loader,
    input_dim, output_dim,
    seq_len, pred_len,
    hidden_dim, num_layers,
    lr, epochs=1, patience=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_type='lstm'
):
    """
    Trains a time series forecasting model.
    """
    model_type = model_type.lower()

    # 1) LSTM
    if model_type == 'lstm':
        model = TimeSeriesLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)

    # 2) Transformer
    elif model_type == 'transformer':
        model = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=hidden_dim,
            n_heads=8,
            d_ff=2048,
            num_layers=num_layers,
            dropout=0.05
        ).to(device)

    # 3) Autoformer
    elif model_type == 'autoformer':
        model = Autoformer(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=hidden_dim,
            n_heads=8,
            d_ff=2048,
            num_layers=num_layers,
            dropout=0.05,
            factor=3
        ).to(device)

    # 4) iTransformer
    elif model_type == 'itransformer':
        model = iTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=hidden_dim,
            n_heads=8,
            d_ff=128,
            num_layers=num_layers,
            dropout=0.05,
            embed="fixed",
            freq="h",
            factor=3,
            activation="gelu"
        ).to(device)

    # 5) Informer & FEDformer
    elif model_type == 'informer':
        model = Informer(
            enc_in=input_dim,
            dec_in=input_dim,
            pred_len=pred_len,
            label_len=seq_len // 2,
            d_model=hidden_dim,
            n_heads=8,
            e_layers=num_layers,
            d_layers=1,
            d_ff=2048,
            dropout=0.05,
            activation="gelu",
            distil=False,
            embed="fixed",
            freq="h",
            factor=3
        ).to(device)

    elif model_type == 'fedformer':
        model = FEDformer(
            enc_in=input_dim,
            dec_in=input_dim,
            pred_len=pred_len,
            c_out=output_dim,
            seq_len=seq_len,
            label_len=seq_len // 2,
            d_model=hidden_dim,
            n_heads=8,
            e_layers=num_layers,
            d_layers=1,
            d_ff=2048,
            dropout=0.05,
            activation="gelu",
            distil=False,
            embed="fixed",
            freq="h",
            factor=3
        ).to(device)

    # 6) TimesNet
    elif model_type == 'timesnet':
        label_len   = seq_len
        num_kernels = min(6, seq_len)
        top_k       = min(5, seq_len)
        model = TimesNet(
            enc_in=input_dim,
            c_out=output_dim,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=hidden_dim,
            d_ff=32,
            embed='fixed',
            freq='h',
            e_layers=num_layers,
            dropout=0.05,
            top_k=top_k,
            num_kernels=num_kernels
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs, target = model.prepare_batch(batch)
            output = model(*inputs)
            # LSTM returns full history → keep last pred_len steps
            if model_type == 'lstm':
                output = output[:, -pred_len:, :]
            # Fedformer & TimesNet return one channel per input feature → keep only the target (first) channel
            elif model_type in ('fedformer', 'timesnet', 'itransformer'):
                output = output[..., :1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = tuple(b.to(device) for b in batch)
                inputs, target = model.prepare_batch(batch)
                output = model(*inputs)
                # LSTM returns full history → keep last pred_len steps
                if model_type == 'lstm':
                    output = output[:, -pred_len:, :]
                # Fedformer & TimesNet return one channel per input feature → keep only the target (first) channel
                elif model_type in ('fedformer', 'timesnet'):
                    output = output[..., :1]
                valid_loss += criterion(output, target).item()


        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best Valid Loss: {best_valid_loss:.4f}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Best model restored.")

    return model
