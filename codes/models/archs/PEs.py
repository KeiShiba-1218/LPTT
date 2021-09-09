import math
import torch
from torch import nn

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, max_size=(512, 512)):
        super().__init__()
        assert dim%4==0, 'dim must be multiple of 4'
        max_h, max_w = max_size
        self.pe = torch.zeros(dim, max_h, max_w)
        dim = dim//2
        pos_h = torch.arange(max_h).unsqueeze(1)
        pos_w = torch.arange(max_w).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (math.log(10000.0)/dim))
        
        self.pe[0:dim:2, :, :] = torch.sin(pos_w*div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        self.pe[1:dim:2, :, :] = torch.cos(pos_w*div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        self.pe[dim::2, :, :] = torch.sin(pos_h*div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        self.pe[dim+1::2, :, :] = torch.cos(pos_h*div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
    
    def forward(self, x):
        _, _, h, w = x.shape
        # model.to(device) ではself.peはdeviceに送られないみたいなのでforwardの際に送る
        return x + self.pe[:, :h, :w].to(x.device)

class ConditionalPositionalEncoding(nn.Module):
    def __init__(self, dim, k=3, d=1):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=k, stride=1, padding=k//2, dilation=d, groups=dim)
    
    def forward(self, x):
        assert len(x.shape)==4, 'Input shape must be (B, C, H, W)'
        return x + self.proj(x)
