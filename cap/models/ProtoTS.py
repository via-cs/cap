import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.base import BaseTimeSeriesModel

class ProtoTS(BaseTimeSeriesModel):
    """
    ProtoTS
    """
    def __init__(self, seq_len, pred_len, enc_in, d_model, n_prototypes, d_bottle):
        super(ProtoTS, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.n_prototypes = n_prototypes
        self.d_bottle = d_bottle

        self.cont_embeddings = nn.ModuleList([
            nn.Linear(1, self.d_model) for _ in range(self.enc_in)
        ])

        #currently don't consider categorical features
        #total_vars = self.enc_in + self.num_categorical
        total_vars = self.enc_in
        self.fusion = nn.Sequential(
            nn.Linear(total_vars * self.d_model, self.d_bottle),
            nn.ReLU(),
            nn.Linear(self.d_bottle, self.d_model)
        )

        self.encoder = nn.GRU(
            input_size = self.d_model,
            hidden_size = self.d_model,
            batch_first = True
        )

        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, self.d_model))
        self.prototype_patterns = nn.Parameter(torch.randn(self.n_prototypes, self.pred_len))

    def prepare_batch(self, batch):
        x_mark_enc = None
        x_mark_dec = None
        x_dec = None
        X, Y = batch
        x_enc = X
        return (x_enc, x_mark_enc, x_dec, x_mark_dec), Y
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, N = x_enc.shape

        embeds = []
        for i in range(self.enc_in):
            val = x_enc[:, :, i].unsqueeze(-1)
            embeds.append(self.cont_embeddings[i](val))
        
        x_embed = torch.stack(embeds, dim=2)

        x_flat = x_embed.view(B, L, -1)
        x_fused = self.fusion(x_flat)
        _, h_n = self.encoder(x_fused)
        z = h_n[-1]

        distances = torch.cdist(z.unsqueeze(1), self.prototypes.unsqueeze(0)).squeeze(1)
        weights = F.softmax(-distances, dim=1)
        prediction = torch.matmul(weights, self.prototype_patterns)

        return prediction
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out.unsqueeze(-1)