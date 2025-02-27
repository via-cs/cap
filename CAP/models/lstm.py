import torch
import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
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

        # LSTM layer
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False, dropout=dropout)
        # RNN layer
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=False, dropout=dropout, nonlinearity='tanh')


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
        # lstm_out, _ = self.lstm(x)  # LSTM output: (seq_len, batch_size, hidden_dim)
        h_0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_dim, device=x.device)  # Initialize hidden state

        rnn_out, _ = self.rnn(x, h_0)  # (seq_len, batch_size, hidden_dim)
        output = self.fc(rnn_out)  # Fully connected layer: (seq_len, batch_size, output_dim)
        return output
