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

from utils.layers import PositionalEncoding, Head, EncoderBlock

class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=32, 
                 in_channels=3, 
                 emb_size=128, 
                 num_heads=8, 
                 num_layers=6, 
                 expansion=4, 
                 dropout=0.1, 
                 num_classes=80):
        
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(in_channels = in_channels,
                                         out_channels= emb_size,
                                         kernel_size = patch_size,
                                         stride = patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_encoding = PositionalEncoding(emb_size, max_len = self.num_patches + 1)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(emb_size, num_heads, dropout, expansion) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(emb_size, num_classes)
         
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)                                       # (B, emb_size, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)                                      # (B, num_patches, emb_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)                              # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)                                       # (B, num_patches + 1, emb_size)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        cls = x[:, 0]                                     # (B, emb_size) — extract CLS token
        out = self.classifier(cls)                      # (B, num_classes)
        return out