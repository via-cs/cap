import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from ..layers.SelfAttention_Family import ProbAttention, AttentionLayer
from ..layers.Embed import DataEmbedding
from ..utils.base import BaseTimeSeriesModel


class Informer(BaseTimeSeriesModel):
    """
    Informer model optimized for forecasting (long & short-term).
    """

    def __init__(self, enc_in, dec_in, pred_len, label_len, d_model=512, embed="fixed", freq="h", dropout=0.1,
                 factor=5, n_heads=8, e_layers=2, d_layers=1, d_ff=1024, activation="gelu", distil=False):
        super(Informer, self).__init__()
        
        self.pred_len = pred_len
        self.label_len = label_len

        # Embedding layers
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, 1, bias=True)  # Output shape: (B, seq, 1)
        )

    def prepare_batch(self, batch):
        """
        Prepare encoder/decoder inputs for Informer.
        """
        X, Y = batch  # X: [B, seq_len, in_dim], Y: [B, pred_len, out_dim]
        label_len = self.label_len
        pred_len = self.pred_len

        # x_enc: raw input
        x_enc = X

        # decoder input: last label_len of the target + zeros
        dec_hist = X[:, -label_len:, :]  # [B, label_len, 7]
        dec_pad = torch.zeros(X.size(0), pred_len, X.size(2)).to(X.device)  # [B, pred_len, 7]
        x_dec = torch.cat([dec_hist, dec_pad], dim=1)  # [B, label_len + pred_len, 7]


        # Time features can be zeros
        x_mark_enc = torch.zeros_like(x_enc)
        x_mark_dec = torch.zeros_like(x_dec)

        return (x_enc, x_mark_enc, x_dec, x_mark_dec), Y


    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out[:, -self.pred_len:, :]  # Shape: (B, pred_len, 1)

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # Normalization
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - mean_enc) / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = dec_out * std_enc + mean_enc  # De-normalization
        return dec_out[:, -self.pred_len:, :]  # Shape: (B, pred_len, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, task="long"):
        if task == "long":
            return self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif task == "short":
            return self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise ValueError("Task must be either 'long' or 'short'.")
