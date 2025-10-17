"""
Training utilities for time series models:
train_loader (DataLoader): training dataset loader yielding batches.
valid_loader (DataLoader): validation dataset loader.
input_dim (int): number of features in the input sequences.
output_dim (int): number of features in the output (forecast) sequences.
seq_len (int): length of input sequence window.
pred_len (int): length of prediction horizon.
d_model (int): model dimension (used for transformer-based models and LSTM hidden dimension).
num_layers (int): number of layers in the model (if applicable).
epochs (int): training epochs.
lr (float): learning rate for the optimizer.
patience (int): early stopping patience.
device (str): device to train on ("cpu" or "cuda").
model_type (str): identifier of which model to train (e.g. "lstm", "transformer", etc.).
d_ff (int): feed-forward dimension for transformer-based models.
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
from ..models.TimeXer import TimeXer
from ..models.TimeMixer import TimeMixer
from ..models.PatchTST import PatchTST
from ..models.DSFormer import DSFormer
from ..models.SimpleTM import SimpleTM
from ..models.Crossformer import Crossformer
from ..models.DLinear import DLinear
from ..models.TimeLLM import TimeLLM

def train_model(
    train_loader, valid_loader,
    input_dim, output_dim,
    seq_len, pred_len,
    d_model, num_layers,
    lr, epochs=1, patience=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_type='lstm',
    d_ff=2048,
    loss_metric='mse'
):
    """
    Trains a time series forecasting model.
    """
    model_type = model_type.lower()

    # 1) LSTM
    if model_type == 'lstm':
        model = TimeSeriesLSTM(input_dim, d_model, output_dim, num_layers).to(device)

    # 2) Transformer
    elif model_type == 'transformer':
        model = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=8,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=0.1
        ).to(device)

    # 3) Autoformer
    elif model_type == 'autoformer':
        model = Autoformer(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=8,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=0.1,
            factor=3
        ).to(device)

    # 4) iTransformer
    elif model_type == 'itransformer':
        model = iTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=8,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=0.1,
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
            d_model=d_model,
            n_heads=8,
            e_layers=num_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=0.1,
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
            d_model=d_model,
            n_heads=8,
            e_layers=num_layers,
            d_layers=1,
            d_ff=d_ff,
            dropout=0.1,
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
            d_model=d_model,
            d_ff=d_ff,
            embed='fixed',
            freq='h',
            e_layers=num_layers,
            dropout=0.1,
            top_k=top_k,
            num_kernels=num_kernels
        ).to(device)

    # 7) TimeXer
    elif model_type == 'timexer':
        model = TimeXer(
            enc_in=input_dim, 
            seq_len=seq_len, 
            pred_len=pred_len, 
            use_norm = True,
            patch_len=16,
            d_ff=d_ff, 
            activation='gelu', 
            num_layers=num_layers, 
            n_heads=8, 
            d_model=d_model, 
            dropout=0.1, 
            factor=3
        ).to(device)
    
    # 8) TimeMixer
    elif model_type == 'timemixer':
        model = TimeMixer(
            seq_len=seq_len,
            label_len=0, 
            pred_len=pred_len, 
            down_sampling_window=2, 
            channel_independence=True, 
            num_layers=num_layers,
            moving_avg=25, 
            enc_in=input_dim, 
            d_model=d_model,
            d_ff=d_ff, 
            embed='fixed', 
            freq='h', 
            dropout=0.1, 
            use_norm=1, 
            down_sampling_layers=3, 
            c_out=input_dim, 
            down_sampling_method='avg',
            decomp_method='moving_avg',
            top_k=5
        ).to(device)
    
    # 9) PatchTST
    elif model_type == 'patchtst':
        model = PatchTST(
            seq_len=seq_len, 
            pred_len=pred_len, 
            d_model=d_model, 
            dropout=0.1, 
            factor=3, 
            n_heads=16, 
            d_ff=d_ff,
            activation='gelu', 
            num_layers=num_layers, 
            enc_in=input_dim, 
            patch_len=16, 
            stride=8
        ).to(device)
    
    # 10) DSFormer
    elif model_type == 'dsformer':
        model = DSFormer(
            seq_len=seq_len, 
            pred_len=pred_len, 
            n_vars=7, 
            num_layers=num_layers, 
            dropout=0.15, 
            muti_head=1, 
            num_samp=3, 
            IF_node=True
        ).to(device)

    # 11) SimpleTM
    elif model_type == 'simpletm':
        model = SimpleTM(
            seq_len=seq_len, 
            pred_len=pred_len,
            d_ff=d_ff, 
            d_model=d_model, 
            dropout=0.1,
            num_layers=num_layers, 
            factor=1, 
            dec_in=7
        ).to(device)

    # 12) Crossformer
    elif model_type == 'crossformer':
        model = Crossformer(
            enc_in=input_dim, 
            seq_len=seq_len, 
            pred_len=pred_len,
            num_layers=num_layers, 
            d_model=d_model, 
            n_heads=4, 
            d_ff=d_ff, 
            dropout=0.1, 
            factor=3
        ).to(device)

    # 13) DLinear
    elif model_type == 'dlinear':
        model = DLinear(
            seq_len=seq_len, 
            pred_len=pred_len, 
            moving_avg=25, 
            enc_in=input_dim
        ).to(device)
    
    # 14) TimeLLM
    elif model_type == 'timellm':
        model = TimeLLM(
            pred_len=pred_len, 
            seq_len=seq_len, 
            enc_in=input_dim, 
            d_ff=d_ff, 
            top_k=5, 
            llm_dim=4086, 
            patch_len=16, 
            stride=8, 
            llm_model='LLAMA', 
            llm_layers=6, 
            prompt_domain=0, 
            content='Weather is recorded every 10 minutes for the 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc.', 
            dropout=0.1, 
            d_model=d_model, 
            n_heads=8
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Training setup
    if loss_metric == 'mse':
        criterion = nn.MSELoss()
    elif loss_metric == 'mae':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss metric: {loss_metric}")
    
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
            #print("inputs shape:", inputs.shape)
            output = model(*inputs)
            # LSTM returns full history → keep last pred_len steps
            if model_type == 'lstm':
                output = output[:, -pred_len:, :]
            # Fedformer & TimesNet return one channel per input feature → keep only the target (first) channel
            #elif model_type in ('fedformer', 'timesnet', 'itransformer'):
            if output.shape[-1] > 1:
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
                #elif model_type in ('fedformer', 'timesnet'):
                if output.shape[-1] > 1:
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
