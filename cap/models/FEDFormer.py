import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Embed import DataEmbedding
from ..layers.AutoCorrelation import AutoCorrelationLayer
from ..layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from ..layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from ..layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class FEDformer(nn.Module):
    """
    FEDformer: Frequency-domain Transformer for Long Sequence Time-Series Forecasting
    Paper: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, enc_in, dec_in, pred_len, c_out, seq_len, label_len,
                 d_model=512, embed="fixed", freq="h", dropout=0.1, factor=5,
                 n_heads=8, e_layers=2, d_layers=1, d_ff=1024, activation="gelu",
                 moving_avg=25, distil=False, version='fourier', mode_select='random', modes=32):
        """
        Initializes the FEDformer model.
        
        Args:
            enc_in (int): Number of encoder input features.
            dec_in (int): Number of decoder input features.
            pred_len (int): Length of forecast horizon.
            c_out (int): Number of output channels (1 for univariate forecasting).
            seq_len (int): Length of input sequence.
            label_len (int): Length of the known portion in decoder input.
            d_model (int): Dimension of the model.
            embed (str): Embedding type ('fixed' or 'learnable').
            freq (str): Frequency of timestamps.
            dropout (float): Dropout rate.
            factor (int): Attention factor.
            n_heads (int): Number of attention heads.
            e_layers (int): Number of encoder layers.
            d_layers (int): Number of decoder layers.
            d_ff (int): Feedforward layer dimension.
            activation (str): Activation function ('gelu' or 'relu').
            moving_avg (int): Moving average window for trend decomposition.
            distil (bool): Whether to use distillation.
            version (str): 'fourier' or 'wavelets' for frequency transformation.
            mode_select (str): Mode selection method ('random' or 'low').
            modes (int): Number of modes to be selected.
        """
        super(FEDformer, self).__init__()

        self.pred_len = pred_len
        self.label_len = label_len
        self.seq_len = seq_len

        # Decomposition for time series
        self.decomp = series_decomp(moving_avg)

        # Embeddings
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Choose frequency-based attention mechanism
        if version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(
                in_channels=d_model, out_channels=d_model, seq_len_q=seq_len // 2 + pred_len,
                seq_len_kv=seq_len, modes=modes, ich=d_model, base='legendre', activation='tanh'
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=d_model, out_channels=d_model, n_heads=n_heads, seq_len=seq_len,
                modes=modes, mode_select_method=mode_select
            )
            decoder_self_att = FourierBlock(
                in_channels=d_model, out_channels=d_model, n_heads=n_heads,
                seq_len=seq_len // 2 + pred_len, modes=modes, mode_select_method=mode_select
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=d_model, out_channels=d_model, seq_len_q=seq_len // 2 + pred_len,
                seq_len_kv=seq_len, modes=modes, mode_select_method=mode_select, num_heads=n_heads
            )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, d_model, n_heads),
                    d_model, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, d_model, n_heads),
                    AutoCorrelationLayer(decoder_cross_att, d_model, n_heads),
                    d_model, c_out, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation
                )
                for _ in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasts the next `pred_len` values given input sequences.
        """
        # Decomposition
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Prepare decoder inputs
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Decoder
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # Final output
        dec_out = trend_part + seasonal_part
        return dec_out[:, -self.pred_len:, :]  # Shape: (batch, pred_len, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for forecasting.
        """
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
