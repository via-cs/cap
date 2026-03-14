import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..layers.Embed import DataEmbedding_inverted
import numpy as np


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for Time-Series Forecasting
    Paper link: https://arxiv.org/abs/2310.06625
    Compatible with CAP framework
    """

    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, d_ff=2048, 
                 num_layers=2, dropout=0.05, embed="fixed", freq="h", factor=5, activation="gelu"):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            seq_len (int): Input sequence length.
            pred_len (int): Number of future steps to forecast.
            d_model (int): Dimension of model embeddings.
            n_heads (int): Number of attention heads.
            d_ff (int): Feedforward network dimension.
            num_layers (int): Number of encoder layers.
            dropout (float): Dropout rate.
            embed (str): Embedding type ('fixed' or 'learnable').
            freq (str): Frequency encoding for timestamps.
            factor (int): Attention factor.
            activation (str): Activation function.
        """
        super(iTransformer, self).__init__()
        print(f"iTransformer initialized with d_model: {d_model}, n_heads: {n_heads}, d_ff: {d_ff}, num_layers: {num_layers}, dropout: {dropout}")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Projection layer for forecasting
        self.projection = nn.Linear(d_model, pred_len, bias=True)

    def prepare_batch(self, batch):
        """
        Prepares (X, Y) batch for iTransformer. Returns (X,), Y so model(*X) == model(X[0])
        Compatible with CAP framework
        """
        X, Y = batch
        return (X,), Y

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasts the next `pred_len` values given input sequences.
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for long-term forecasting.
        """
        # Handle case where only x_enc is provided (single input)
        if x_mark_enc is None:
            x_mark_enc = torch.zeros_like(x_enc)
        if x_dec is None:
            x_dec = torch.zeros_like(x_enc)
        if x_mark_dec is None:
            x_mark_dec = torch.zeros_like(x_enc)
            
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        #return dec_out[:, -self.pred_len:, 0].unsqueeze(-1)  # [B, L, 1] - only first target
        return dec_out[:, -self.pred_len:, :]