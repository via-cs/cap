import torch
from torch import nn, optim
import torch.nn.functional as F
from ..layers.embed_block import embed
from ..layers.TVA_block import TVA_block_att
from ..layers.decoder_block import TVADE_block
from ..layers.revin import RevIN
from ..utils.base import BaseTimeSeriesModel

class DSFormer(BaseTimeSeriesModel):
    def __init__(self, seq_len, pred_len, n_vars, num_layers, dropout, muti_head, num_samp, IF_node):
        """
        seq_len: History length
        pred_len：future length
        n_vars：number of variables
        num_layers：number of layer. 1 or 2
        muti_head：number of muti_head attention. 1 to 4
        dropout：dropout. 0.15 to 0.3
        num_samp：muti_head subsequence. 2 or 3
        IF_node:Whether to use node embedding. True or False
        """
        super(DSFormer, self).__init__()

        if IF_node:
            self.inputlen = 2 * seq_len // num_samp
        else:
            self.inputlen = seq_len // num_samp

        ### embed and encoder
        self.RevIN = RevIN(n_vars)
        self.embed_layer = embed(seq_len,n_vars,num_samp,IF_node)
        self.encoder = TVA_block_att(self.inputlen,n_vars,num_layers,dropout, muti_head,num_samp)
        self.laynorm = nn.LayerNorm([self.inputlen])

        ### decorder
        self.decoder = TVADE_block(self.inputlen, n_vars, dropout, muti_head)
        self.output = nn.Conv1d(in_channels = self.inputlen, out_channels=pred_len, kernel_size=1)
    
    def prepare_batch(self, batch):
        X, Y = batch
        x_enc = X
        x_mark_enc = None
        x_dec = None
        x_mark_dec = None
        return (x_enc, x_mark_enc, x_dec, x_mark_dec), Y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Input [B,H,N]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N]: B is batch size. N is the number of variables. L is the future length

        ### embed
        x = self.RevIN(x_enc,'norm').transpose(-2,-1)
        x_1, x_2 = self.embed_layer(x)

        ### encoder
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x = x_1 + x_2
        x = self.laynorm(x)

        ### decorder
        x = self.decoder(x)
        x = self.output(x.transpose(-2,-1))
        x = self.RevIN(x, 'denorm')

        return x
