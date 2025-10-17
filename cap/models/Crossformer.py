import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from ..layers.Embed import PatchEmbedding
from ..layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from ..models.PatchTST import FlattenHead
from types import SimpleNamespace
from ..utils.base import BaseTimeSeriesModel

from math import ceil


class Crossformer(BaseTimeSeriesModel):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, enc_in, seq_len, pred_len, num_layers, d_model, n_heads, d_ff, dropout, factor):
        super(Crossformer, self).__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seg_len = 12
        self.win_size = 2

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (num_layers - 1)))
        self.head_nf = d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(d_model, self.seg_len, self.seg_len, self.pad_in_len - seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, enc_in, self.in_seg_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        configs = SimpleNamespace(
            factor = factor,
            dropout = dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l is 0 else self.win_size, d_model, n_heads, d_ff,
                            1, dropout,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), factor
                            ) for l in range(num_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, enc_in, (self.pad_out_len // self.seg_len), d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), factor, d_model, n_heads,
                                           d_ff, dropout),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    self.seg_len,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    # activation=configs.activation,
                )
                for _ in range(num_layers + 1)
            ],
        )


    def prepare_batch(self, batch):
        X, Y = batch
        x_enc = X
        x_mark_enc = None
        x_dec = None
        x_mark_dec = None
        return (x_enc, x_mark_enc, x_dec, x_mark_dec), Y


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]