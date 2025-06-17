import torch
import torch.nn as nn
from ..layers.Embed import DataEmbedding
from ..utils.base import BaseTimeSeriesModel

class TimeSeriesLSTM(BaseTimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        """
        LSTM model for time-series forecasting.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in LSTM.
            output_dim (int): Number of output features.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer with batch_first=True to accept input of shape (batch_size, seq_len, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def prepare_batch(self, batch):
        """
        Prepares (X, Y) batch for LSTM. Returns (X, Y) tuple so that model(*inputs) == model(X).
        """
        X, Y = batch
        return (X,), Y

    def forward(self, x):
        """
        Forward pass of LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        # Initialize hidden state and cell state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h_0, c_0))  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Apply fully connected layer to each time step
        output = self.fc(lstm_out)  # (batch_size, seq_len, output_dim)
        return output
