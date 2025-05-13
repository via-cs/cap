import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from ..layers.PositionalEncoding import PositionalEncoding
from typing import List, Type, Dict, Any
import importlib
from .Autoformer import Autoformer
from .FEDFormer import FEDformer
from .Informer import Informer
from .transformer import Transformer
from .lstm import TimeSeriesLSTM

def available_models():
    return {
        'transformer': Transformer,
        'lstm': TimeSeriesLSTM,
        'informer': Informer,
        'autoformer': Autoformer,
        'fedformer': FEDformer
    }

class ManagerModel(nn.Module):
    """
    Manager model that selects the most appropriate worker model for each time series.
    """
    def __init__(self, input_dim: int, worker_count: int, d_model: int = 512, 
                 n_heads: int = 8, d_ff: int = 2048, num_layers: int = 3, dropout: float = 0.1):
        super(ManagerModel, self).__init__()
        self.model_type = 'Manager'
        
        # 输入特征编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # 使用 GELU 激活函数
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, worker_count)
        )
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """使用更好的权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, src):
        """
        前向传播
        
        Args:
            src: 输入张量，形状为 (batch_size, seq_len, input_dim)
            
        Returns:
            工人模型概率分布，形状为 (batch_size, worker_count)
        """
        # 编码输入
        output = self.encoder(src)
        
        # 通过 Transformer
        output = self.transformer_encoder(output)
        
        # 使用最后一个时间步的编码
        output = output[:, -1, :]
        
        # 预测工人概率
        output = self.decoder(output)
        
        # 使用温度缩放和 softmax
        temperature = 1.0
        output = F.softmax(output / temperature, dim=-1)
        
        return output

class WorkerWrapper(nn.Module):
    """
    A wrapper class for worker models that standardizes their interface.
    """
    def __init__(self, model_class: Type[nn.Module], model_args: Dict[str, Any]):
        super(WorkerWrapper, self).__init__()
        self.model = model_class(**model_args)
        self.model_type = 'Worker'
        # Store pred_len from model args
        self.pred_len = model_args.get('pred_len', None)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Standardized forward pass that handles different model interfaces.
        
        Args:
            x_enc: Input tensor
            x_mark_enc: Optional temporal features for encoder
            x_dec: Optional decoder input
            x_mark_dec: Optional temporal features for decoder
            
        Returns:
            Model predictions
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Handle models that require all four arguments
        if isinstance(self.model, (Autoformer, FEDformer, Informer)):
            # For these models, we need to create dummy tensors if not provided
            if x_mark_enc is None:
                x_mark_enc = torch.zeros_like(x_enc)
            if x_dec is None:
                # Use pred_len from model args or default to half of input length
                pred_len = self.pred_len if self.pred_len is not None else x_enc.shape[1] // 2
                x_dec = torch.zeros((x_enc.shape[0], pred_len, x_enc.shape[2]), device=x_enc.device)
            if x_mark_dec is None:
                x_mark_dec = torch.zeros_like(x_dec)
            
            # For Autoformer, ensure x_dec has same feature dimension as x_enc
            if isinstance(self.model, Autoformer):
                x_dec = x_dec[:, :, :1]  # Keep only first feature dimension
            
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Ensure output has same sequence length as target
            if output.shape[1] != self.pred_len:
                output = output[:, :self.pred_len, :]
            
            return output
        
        # For simpler models like LSTM and Transformer
        output = self.model(x_enc)
        
        # Ensure output has same sequence length as target
        if output.shape[1] != self.pred_len:
            output = output[:, :self.pred_len, :]
        
        return output

def create_worker_pool(model_configs: List[Dict[str, Any]], available_models: Dict[str, Type[nn.Module]]) -> List[WorkerWrapper]:
    """
    Creates a pool of worker models based on configuration.
    
    Args:
        model_configs: List of model configurations
        available_models: Dictionary mapping model names to model classes
        
    Returns:
        List of wrapped worker models
    """
    workers = []
    for config in model_configs:
        model_name = config.pop('model_name')
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not found in available models")
        
        model_class = available_models[model_name]
        workers.append(WorkerWrapper(model_class, config))
    
    return workers

# Example usage:
# available_models = {
#     'transformer': Transformer,
#     'lstm': LSTM,
#     'informer': Informer,
#     # ... other models
# }
#
# model_configs = [
#     {'model_name': 'transformer', 'input_dim': 10, 'output_dim': 1, ...},
#     {'model_name': 'lstm', 'input_dim': 10, 'output_dim': 1, ...},
#     # ... configs for other workers
# ]
#
# worker_pool = create_worker_pool(model_configs, available_models)
# manager = ManagerModel(input_dim=10, worker_count=len(worker_pool))

class WorkerModel(nn.Module):
    def __init__(self, in_dim, out_dim, ninp, nhead, nhid, nlayers, data_types, dropout=0.2):
        super(WorkerModel, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model_type = 'CATP'
        self.ninp = ninp
        self.encoder_input_layer = nn.Linear(in_dim, ninp).to(self.device)
        self.decoder_input_layer = nn.Linear(out_dim, ninp).to(self.device)
        self.linear_mapping = nn.Linear(ninp, out_dim).to(self.device)
        self.normalize_layer = nn.Sigmoid().to(self.device)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        
        encoder_layer = TransformerEncoderLayer(ninp, nhead, nhid, dropout).to(self.device)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers).to(self.device)
        decoder_layer = TransformerDecoderLayer(ninp, nhead, nhid, dropout).to(self.device)
        self.transformer_decoder = TransformerDecoder(decoder_layer, nlayers).to(self.device)

        self.src_mask = None
        self.tgt_mask = None
        
        # Register coordinate normalization parameters
        self.register_buffer("min_coord", torch.tensor([8240, 8220], dtype=torch.float32))
        self.register_buffer("max_coord", torch.tensor([24510, 24450], dtype=torch.float32))
        self.to(self.device)

        self.init_weights()

    def generate_square_subsequent_mask(self, dim1, dim2):
        return torch.triu(torch.ones(dim1, dim2, device=self.device) * float('-inf'), diagonal=1)

    def init_weights(self):
        nn.init.xavier_uniform_(self.encoder_input_layer.weight)
        nn.init.zeros_(self.encoder_input_layer.bias)
        nn.init.xavier_uniform_(self.decoder_input_layer.weight)
        nn.init.zeros_(self.decoder_input_layer.bias)
        nn.init.xavier_uniform_(self.linear_mapping.weight)
        nn.init.zeros_(self.linear_mapping.bias)

    def normalize_coord(self, coord):
        norm_coord = torch.zeros_like(coord, device=self.device)
        norm_coord[:, :, 0::2] = (coord[:, :, 0::2] - self.min_coord[0]) / (self.max_coord[0] - self.min_coord[0])
        norm_coord[:, :, 1::2] = (coord[:, :, 1::2] - self.min_coord[1]) / (self.max_coord[1] - self.min_coord[1])
        return norm_coord

    def restore_coord(self, coord):
        res_coord = coord * (self.max_coord - self.min_coord) + self.min_coord
        return res_coord

    def forward(self, src, tgt, role, test=False, has_mask=True):
        # Normalize and encode source sequence
        mem = self.normalize_coord(src).to(self.device)
        mem = self.encoder_input_layer(mem).to(self.device)
        mem = self.pos_encoder(mem)
        mem = self.transformer_encoder(mem)

        # Prepare target sequence
        role = int(role)
        idx = (role // 10) * 5 + role % 10 - 6
        if test:
            if tgt is None:
                output = src[-1, :, idx*2:idx*2+2].unsqueeze(0)
            else:
                output = torch.cat((src[-1, :, idx*2:idx*2+2].unsqueeze(0), tgt), dim=0)
        else:
            output = torch.cat((src[-1, :, idx*2:idx*2+2].unsqueeze(0), tgt[:-1]), dim=0)

        # Generate masks if needed
        if has_mask:
            device = mem.device
            if self.src_mask is None or self.src_mask.size(0) != len(mem):
                self.src_mask = self.generate_square_subsequent_mask(len(output), len(mem)).to(device)
            if self.tgt_mask is None or self.tgt_mask.size(0) != len(mem):
                self.tgt_mask = self.generate_square_subsequent_mask(len(output), len(output)).to(device)
        else:
            self.src_mask = None
            self.tgt_mask = None

        # Decode and process output
        output = self.normalize_coord(output)
        output = self.decoder_input_layer(output)
        output = self.pos_encoder(output)
        output = self.transformer_decoder(
            tgt=output,
            memory=mem,
            tgt_mask=self.tgt_mask,
            memory_mask=self.src_mask
        )
        output = self.linear_mapping(output)
        output = self.normalize_layer(torch.clamp(output, min=-50, max=50))
        output = self.restore_coord(output)

        return output 