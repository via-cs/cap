import torch
import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
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

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, output_dim).
        """
        lstm_out, _ = self.lstm(x)  # LSTM output: (seq_len, batch_size, hidden_dim)
        output = self.fc(lstm_out)  # Fully connected layer: (seq_len, batch_size, output_dim)
        return output
