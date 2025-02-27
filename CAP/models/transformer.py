import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class TimeSeriesTransformer(nn.Module):
    """
    Transformer for Time-Series Forecasting
    """
    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, d_ff=2048, num_layers=3, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Data Embedding (No Time Features)
        self.embedding = DataEmbedding(input_dim, d_model, embed_type='fixed', freq='h', dropout=dropout)

        # Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation='gelu'
                ) for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Output Layer for Forecasting
        self.projection = nn.Linear(d_model, output_dim)

    def forward(self, x_enc):
        """
        Forward pass for time-series forecasting.

        Args:
            x_enc: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Predicted output tensor of shape (batch_size, pred_len, output_dim)
        """
        enc_out = self.embedding(x_enc)  # (batch_size, seq_len, d_model)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # (batch_size, seq_len, d_model)

        # Use only the last timestep's encoding for prediction
        forecast_out = self.projection(enc_out[:, -self.pred_len:, :])  # (batch_size, pred_len, output_dim)

        return forecast_out
