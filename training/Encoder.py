import torch
import torch.nn as nn
from Function.Function import *
from training.PositionalEncoding import*

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.relu(self.w_1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,h,dropout):
        super(EncoderLayer, self).__init__()
        # —————————————— # 自注意力
        self.SelfMhaLayerNorm = nn_LayerNorm(d_model)
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model,num_heads=h,dropout=dropout,batch_first=True)
        # —————————————— # 全连接
        self.FFNLayerNorm = nn_LayerNorm(d_model)
        self.FFN = FFN(d_model,d_ff)
        # —————————————— # dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self,src,src_pad_mask):
        # —————————————— # 自注意力
        src_ = self.SelfMhaLayerNorm(src)
        src_ = self.SelfMha(src_,src_,src_,key_padding_mask=src_pad_mask,need_weights=False)[0]
        src  = src + self.dropout(src_)
        # —————————————— # 全连接
        src_ = self.FFNLayerNorm(src)
        src_ = self.FFN(src_)
        src  = src + self.dropout(src_)
        return src

class Encoder(nn.Module):
    def __init__ (self,d_model,d_ff,h,N,vocab_size,dropout,ProcessId):
        super(Encoder, self).__init__()
        # —————————————— # 实例化
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positionalencoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model,d_ff,h,dropout) for _ in range(N)])
        self.EndLayerNorm = nn_LayerNorm(d_model)
        # —————————————— # 初始化参数
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(f'cuda:{ProcessId}')
    def forward(self,src,src_pad_mask):
        # —————————————— # 嵌入
        src = self.positionalencoding(self.tok_embedding(src)*self.scale)
        # —————————————— # 前向传播
        for layer in self.layers:
            src = layer(src,src_pad_mask)
        # —————————————— # 输出
        src = self.EndLayerNorm(src)
        return src





