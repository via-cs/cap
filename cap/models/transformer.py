import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..layers.Embed import DataEmbedding
import numpy as np


class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity for Time-Series Forecasting
    Compatible with CAP framework
    """
    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, d_ff=2048, num_layers=2, dropout=0.05):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = False

        # Data Embedding (No Time Features)
        self.enc_embedding = DataEmbedding(input_dim, d_model, embed_type='fixed', freq='h', dropout=dropout)
        self.dec_embedding = DataEmbedding(input_dim, d_model, embed_type='fixed', freq='h', dropout=dropout)

        # Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout, output_attention=self.output_attention),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation='gelu'
                ) for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Transformer Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor=5, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation='gelu',
                )
                for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, output_dim, bias=True)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass for time-series forecasting.
        Compatible with CAP framework - can handle both single input and full encoder-decoder inputs.

        Args:
            x_enc: Input tensor of shape (batch_size, seq_len, input_dim)
            x_mark_enc: Optional temporal features for encoder (not used)
            x_dec: Optional decoder input (if None, will be created from x_enc)
            x_mark_dec: Optional temporal features for decoder (not used)
            enc_self_mask: Optional mask for encoder self-attention
            dec_self_mask: Optional mask for decoder self-attention
            dec_enc_mask: Optional mask for decoder-encoder cross-attention

        Returns:
            Predicted output tensor of shape (batch_size, pred_len, output_dim)
        """
        # Create decoder input if not provided
        if x_dec is None:
            # Use the last timestep of encoder input as decoder input
            x_dec = x_enc[:, -1:, :].repeat(1, self.pred_len, 1)
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    def prepare_batch(self, batch):
        """
        Prepares (X, Y) batch for Transformer. Returns (X,), Y so model(*X) == model(X[0])
        Compatible with CAP framework
        """
        X, Y = batch
        return (X,), Y 