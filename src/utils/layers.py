import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import os

from PIL import Image
from tqdm import tqdm
from typing import Optional
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- Single Attention Head ---
class Head(nn.Module):
    def __init__(self, emb_size, head_size, dropout=0.0, bias=False):
        super().__init__()
        self.key   = nn.Linear(emb_size, head_size, bias=bias)
        self.query = nn.Linear(emb_size, head_size, bias=bias)
        self.value = nn.Linear(emb_size, head_size, bias=bias)
        self.scale = 1/np.sqrt(head_size)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, q, k, v, mask=None):
        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)
        return self.scaled_dot_product_attention(Q, K, V, mask)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = nn.Softmax(dim=-1)(scores)
        self.attn_weights = attn.detach().cpu()
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.0, bias=False):
        super().__init__()
        assert emb_size % num_heads == 0
        
        head_size = emb_size // num_heads
        
        self.heads = nn.ModuleList([
            Head(emb_size, head_size, dropout, bias) for _ in range(num_heads)
        ])
        
        self.linear = nn.Linear(emb_size, emb_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        out = torch.cat([h(q, k, v, mask) for h in self.heads], dim = -1)
        out = self.linear(out)
        out = self.dropout(out)
        return out

    def get_attention_maps(self):
        return [h.attn_weights for h in self.heads if h.attn_weights is not None]

# --- Encoder Block ---
class EncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.0, expansion=4):
        super().__init__()
        self.attn = MultiHeadAttention(emb_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Linear(expansion * emb_size, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# --- Decoder Block ---
class DecoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.0, expansion=4, use_cross_attn=False):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(emb_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        
        
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = MultiHeadAttention(emb_size, num_heads, dropout)
            self.norm2 = nn.LayerNorm(emb_size)
        else:
            self.cross_attn = None
            self.norm2 = None
        
        self.ff = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Linear(expansion * emb_size, emb_size)
        )
        self.norm3 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        enc_out: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        if enc_out is not None:
            if self.cross_attn is None:
                raise ValueError("Cross-attention is not enabled in this DecoderBlock.")
            
            cross_attn_out = self.cross_attn(x, enc_out, enc_out)
            x = self.norm2(x + self.dropout(cross_attn_out))
        
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

# --- Causal Mask Generation ---
def make_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)