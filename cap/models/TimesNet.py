import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from ..layers.Embed import DataEmbedding
from ..layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # Parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # Padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # Reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1D variation to 2D variation
            out = self.conv(out)
            # Reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # Adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # Residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    TimesNet Model: Temporal Convolutional Network for Time-Series Forecasting
    Paper: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, enc_in, seq_len, label_len, pred_len, c_out, d_model=512, embed="fixed", freq="h",
                 dropout=0.1, d_ff=1024, num_kernels=6, top_k=5, e_layers=2):
        """
        Args:
            enc_in (int): Number of input features.
            seq_len (int): Length of input sequence.
            label_len (int): Length of known portion in decoder input.
            pred_len (int): Forecasting horizon.
            c_out (int): Number of output channels.
            d_model (int): Model dimension.
            embed (str): Type of embedding ('fixed' or 'learnable').
            freq (str): Frequency encoding for timestamps.
            dropout (float): Dropout rate.
            d_ff (int): Feedforward layer dimension.
            num_kernels (int): Number of convolution kernels.
            top_k (int): Number of periodic patterns detected.
            e_layers (int): Number of layers in TimesNet.
        """
        super(TimesNet, self).__init__()

        self.pred_len = pred_len
        self.label_len = label_len
        self.seq_len = seq_len

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)

        # TimesBlocks (Stacked Layers)
        self.model = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

        # Forecasting Projection
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def prepare_batch(self, batch):
        """
        Unpack and prepare the batch for the model.
        """
        X, Y = batch
        x_enc = X
        x_mark_enc = torch.zeros_like(X)
        return (x_enc, x_mark_enc), Y
    
    def forecast(self, x_enc, x_mark_enc):
        """
        Forecasts future values using TimesNet.
        """
        # Normalization (from Non-Stationary Transformer)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # Shape: [B, T, C]

        # Align Temporal Dimension
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesNet Processing
        for layer in self.model:
            enc_out = self.layer_norm(layer(enc_out))

        # Projection Back
        dec_out = self.projection(enc_out)

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out[:, -self.pred_len:, :]  # Return only the predicted part

    def forward(self, x_enc, x_mark_enc):
        """
        Forward pass for forecasting.
        """
        return self.forecast(x_enc, x_mark_enc)
