import torch
import torch.nn as nn
from ..layers.Embed import DataEmbedding_wo_pos
from ..layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from ..layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Autoformer for Time-Series Forecasting
    """

    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, d_ff=2048, 
                 num_layers=3, dropout=0.1, moving_avg=25, factor=5, activation="gelu"):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            seq_len (int): Input sequence length.
            pred_len (int): Number of future steps to forecast.
            d_model (int): Dimension of model embeddings.
            n_heads (int): Number of attention heads.
            d_ff (int): Feedforward network dimension.
            num_layers (int): Number of encoder & decoder layers.
            dropout (float): Dropout rate.
            moving_avg (int): Moving average filter size for decomposition.
            factor (int): Factor for AutoCorrelation attention.
            activation (str): Activation function.
        """
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = seq_len // 2  # Middle portion of sequence for decoder input

        # Decomposition for trend-seasonal components
        self.decomp = series_decomp(moving_avg)

        # Embeddings
        self.enc_embedding = DataEmbedding_wo_pos(input_dim, d_model, "fixed", "h", dropout)
        self.dec_embedding = DataEmbedding_wo_pos(output_dim, d_model, "fixed", "h", dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    output_dim,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, output_dim, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Autoformer forward pass for forecasting.

        Args:
            x_enc (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            x_mark_enc (Tensor): (Optional) Time features for encoder.
            x_dec (Tensor): Decoder input tensor (used for autoregressive forecasting).
            x_mark_dec (Tensor): (Optional) Time features for decoder.

        Returns:
            Tensor: Forecasted output of shape (batch_size, pred_len, output_dim).
        """
        # Decomposition: Trend & Seasonal components
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(x_dec.device)


        seasonal_init, trend_init = self.decomp(x_enc)
        # print(f"seasonal_init: {trend_init}")
            #   , trend_init: {trend_init}")
        # print(f"mean: {mean.shape}, zeros: {zeros.shape}")

        # Prepare decoder input (label_len past values + zeros for future)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, 0].unsqueeze(-1), zeros], dim=1)
        # print(f"seasonal_init: {seasonal_init.shape}, trend_init: {trend_init.shape}")

        # Encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        # print(f"enc_out: {enc_out}")

        # Decoding
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # print(f"dec_out: {dec_out.shape}")
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # print(f"seasonal_part+trend: {(seasonal_part+trend_part).shape}, trend_part: {trend_part.shape}")

        # Final output (trend + seasonal)
        return trend_part[:, :, 0].unsqueeze(-1) + seasonal_part
        