import torch
import torch
from torch import nn
from codes.models.archs.LPTT_arch import LPTT
from codes.models.archs.LPTN_arch import LPTN
from einops import rearrange
from einops.layers.torch import Rearrange
import torchinfo
import time
from torchvision.transforms import Resize
import math
from codes.models.archs.axial_attention import AxialAttention, AxialImageTransformer, AxialPositionalEmbedding, PositionalEncoding2D

def lptt_test(): # LPTT動作テスト
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    h, w = 3840, 2160
    # h, w = 256, 256
    x = torch.randn(1, 3, h, w).to(device)
    
    lptt = LPTT(num_high=5).to(device)
    
    torchinfo.summary(lptt, input_data=x)
    
    out = lptt(x)
    print(out.size())

def check_summaries(): # summaryの確認
    lptn = LPTN(num_high=5)
    lptt = LPTT(num_high=5)
    torchinfo.summary(lptn, input_size=(16, 3, 256, 256))
    print('#'*64)
    torchinfo.summary(lptt, input_size=(16, 3, 256, 256))

def check_inference():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    h, w = 3840, 2160
    # h, w = 256, 256
    x = torch.randn(1, 3, h, w).to(device)
    lptn = LPTN(num_high=5).to(device)
    lptt = LPTT(num_high=5).to(device)

    _ = lptn(x)
    _ = lptt(x)
    
    t0 = time.time()
    out1 = lptn(x)
    t1 = time.time()
    out2 = lptt(x)
    t2 = time.time()
    
    print(f'LPTN: {t1-t0:f}, LPTT: {t2-t1:f}')

def transforms_test():
    x = torch.randn(16, 3, 256, 256)
    x = Resize((128, 128))(x)
    print(x.size())

def axial_test():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    res = 32
    
    hidden_dim = 16
    model = nn.Sequential(
        nn.Conv2d(3, hidden_dim, kernel_size=1, bias=False),
        PositionalEncoding2D(hidden_dim),
        AxialAttention(dim=hidden_dim, num_dimensions=2, heads=8, dim_heads=16, dim_index=1, sum_axial_out=True),
        nn.Conv2d(hidden_dim, 3, kernel_size=1, bias=False)
    )
    
    torchinfo.summary(model, input_size=(1, 3, res, res))
    x = torch.randn(1, 3, res, res).to(device)
    model = model.to(device)
    out = model(x)
    print(out.shape)

def pos_test():
    x = torch.zeros(16, 32, 64, 64)
    out = PositionalEncoding2D(dim=32)(x)
    print(out.shape)
    

if __name__=='__main__':
    # check_summaries()
    check_inference()