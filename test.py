import torch
import torch
from torch import nn
from codes.models.archs.LPTT_arch import LPTT
from codes.models.archs.LPTT_paper_arch import LPTTPaper
from codes.models.archs.LPTN_arch import LPTN
from einops import rearrange
from einops.layers.torch import Rearrange
import torchinfo
import time
from torchvision.transforms import Resize
import math
from codes.models.archs.axial_attention import AxialAttention

def lptt_test(): # LPTT動作テスト
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # h, w = 6000, 4000
    # h, w = 3840, 2160
    h, w = 256, 256
    x = torch.randn(1, 3, h, w).to(device)
    
    lptt = LPTT(num_high=3).to(device)
    # lptt = LPTTPaper(num_high=5).to(device)
    
    out = lptt(x)
    print(out.size())

def check_summaries(): # summaryの確認
    lptn = LPTN(num_high=6)
    lptt = LPTT(num_high=6)
    b_size = 1
    h, w = 256, 256
    # h, w = 720, 480 # 480p
    # h, w = 1280, 720 # 720p
    # h, w = 1920, 1080 # 1080p
    # h, w = 2560, 1440 # 2K
    # h, w = 3840, 2160 # 4K
    # h, w = 7680, 4320 # 8K
    # h, w = 8192, 8192 # Super Big Size
    torchinfo.summary(lptn, input_size=(b_size, 3, h, w))
    print('#'*64)
    torchinfo.summary(lptt, input_size=(b_size, 3, h, w))

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
        AxialAttention(dim=hidden_dim, num_dimensions=2, heads=8, dim_heads=16, dim_index=1, sum_axial_out=True),
        nn.Conv2d(hidden_dim, 3, kernel_size=1, bias=False)
    )
    
    torchinfo.summary(model, input_size=(1, 3, res, res))
    x = torch.randn(1, 3, res, res).to(device)
    model = model.to(device)
    out = model(x)
    print(out.shape)
    

if __name__=='__main__':
    # lptt_test()
    check_summaries()

