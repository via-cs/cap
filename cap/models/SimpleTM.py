import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SWTAttention_Family import GeomAttentionLayer, GeomAttention
from ..layers.Embed import DataEmbedding_inverted
from ..utils.base import BaseTimeSeriesModel

class SimpleTM(BaseTimeSeriesModel):
    def __init__(self, seq_len, pred_len, d_ff, d_model, dropout, num_layers, factor, dec_in, activation='gelu', output_attention=False, 
                use_norm=True, geomattn_dropout=0.5, alpha=1, kernel_size=None, requires_grad=True, wv='db4', m=1):
        super(SimpleTM, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.geomattn_dropout = dropout
        self.alpha = alpha
        self.kernel_size = kernel_size

        enc_embedding = DataEmbedding_inverted(seq_len, d_model, 
                                               'fixed', 'h', dropout)
        self.enc_embedding = enc_embedding

        encoder = Encoder(
            [  
                EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False, factor, attention_dropout=dropout, 
                            output_attention=output_attention, alpha=self.alpha
                        ),
                        d_model, 
                        requires_grad=requires_grad, 
                        wv=wv, 
                        m=m, 
                        d_channel=dec_in, 
                        kernel_size=self.kernel_size, 
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(num_layers) 
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder = encoder

        projector = nn.Linear(d_model, self.pred_len, bias=True)
        self.projector = projector

    
    def prepare_batch(self, batch):
        X, Y = batch
        x_enc = X
        x_mark_enc = torch.zeros_like(x_enc)
        x_dec = None
        x_mark_dec = None
        return (x_enc, x_mark_enc, x_dec, x_mark_dec), Y


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # x_enc /= stdev
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        enc_embedding = self.enc_embedding
        encoder = self.encoder
        projector = self.projector
        # Linear Projection             B L N -> B L' (pseudo temporal tokens) N 
        enc_out = enc_embedding(x_enc, x_mark_enc) 

        # SimpleTM Layer                B L' N -> B L' N 
        enc_out, attns = encoder(enc_out, attn_mask=None)

        # Output Projection             B L' N -> B H (Horizon) N
        dec_out = projector(enc_out).permute(0, 2, 1)[:, :, :N] 

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, None, None, None)
        return dec_out 