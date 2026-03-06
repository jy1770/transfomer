import torch
import torch.nn as nn
from Function.Function import *
from testing.PositionalEncoding_ import*

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.relu(self.w_1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,h):
        super(EncoderLayer, self).__init__()
        # —————————————— # 自注意力
        self.SelfMhaLayerNorm = nn_LayerNorm(d_model)
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model,num_heads=h,batch_first=True)
        # —————————————— # 全连接
        self.FFNLayerNorm = nn_LayerNorm(d_model)
        self.FFN = FFN(d_model,d_ff)
    def forward(self,src,src_pad_mask):
        # —————————————— # 自注意力
        src_ = self.SelfMhaLayerNorm(src)
        src_ = self.SelfMha(src_,src_,src_,key_padding_mask=src_pad_mask,need_weights=False)[0]
        src  = src + src_
        # —————————————— # 全连接
        src_ = self.FFNLayerNorm(src)
        src_ = self.FFN(src_)
        src  = src + src_
        return src

class Encoder(nn.Module):
    def __init__ (self,d_model,d_ff,h,N,vocab_size):
        super(Encoder, self).__init__()
        # —————————————— # 实例化
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positionalencoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model,d_ff,h) for _ in range(N)])
        self.EndLayerNorm = nn_LayerNorm(d_model)
        # —————————————— # 初始化参数
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to('cuda:0')
    def forward(self,src,src_pad_mask):
        # —————————————— # 嵌入
        src = self.positionalencoding(self.tok_embedding(src)*self.scale)
        # —————————————— # 前向传播
        for layer in self.layers:
            src = layer(src,src_pad_mask)
        # —————————————— # 输出
        src = self.EndLayerNorm(src)
        return src





