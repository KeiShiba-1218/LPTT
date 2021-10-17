import torch
from torch import nn
from codes.models.archs.LPTT_arch import LPTT
from codes.models.archs.LPTN_arch import LPTN
import torchvision
import glob
from PIL import Image
from tqdm import tqdm
import os
import numpy as np

def inference(source_path='./test_images/source', out_path=None, network_path=None, L=None):
    os.makedirs(out_path, exist_ok=True)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # net = LPTT(num_high=L).to(device)
    net = LPTN(num_high=L).to(device)
    load_net = torch.load(network_path)
    net.load_state_dict(load_net, strict=False)
    
    images = sorted(glob.glob(source_path+'/*.jpg'))
    
    net.eval()
    for imgpath in tqdm(images):
        imgname = os.path.basename(imgpath)
        with Image.open(imgpath) as img:
            img = img.convert('RGB')
            # print(imgname, img.size)
            if img.width*img.height>float('inf'): # 4K: 3840*2160, 8K: 7680*4320, for LPTT(L=3): 5464*3070-1
                print('Estimated to be OOM. So continued.')
                continue
            img = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                img = net(img)
            img = img.detach().squeeze(0).cpu().permute(1,2,0).numpy()
            img = np.clip(img, a_min=0.0, a_max=1.0) # これがないと真っ赤だったり真っ青な部分が出てしまう
            img = (img*255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(out_path, imgname), 'JPEG')
            exit()

configs = {
    'LPTN_L3': ['./test_images/out/LPTN_FiveK_L3', 
    './experiments/pretrained_models/net_g_FiveK_numhigh3.pth'], 
    'LPTN_L4': ['./test_images/out/LPTN_FiveK_L4', 
    'experiments/LPTN_FiveK_L4/models/net_g_latest.pth'], 
    'LPTN_L5': ['./test_images/out/LPTN_FiveK_L5', 
    'experiments/LPTN_FiveK_L5/models/net_g_latest.pth'],
    'LPTN_L6': ['./test_images/out/LPTN_FiveK_L6', 
    'experiments/LPTN_FiveK_L6/models/net_g_latest.pth'],
    'LPTT_L3': ['./test_images/out/LPTT_FiveK_L3_PEG-1k9',
    'experiments/LPTT_FiveK_L3_PEG-1k9/models/net_g_latest.pth'], 
    'LPTT_L4': ['./test_images/out/LPTT_FiveK_L4_PEG-1k9', 
    'experiments/LPTT_FiveK_L4_PEG-1k9/models/net_g_latest.pth'], 
    'LPTT_L5': ['./test_images/out/LPTT_FiveK_L5_PEG-1k5',
    'experiments/LPTT_FiveK_L5_PEG-1k5/models/net_g_latest.pth'], 
    'LPTT_L6': ['./test_images/out/LPTT_FiveK_L6_PEG-1k3', 
    'experiments/LPTT_FiveK_L6_PEG-1k3/models/net_g_latest.pth']
}

if __name__=='__main__':
    config = configs['LPTN_L6']
    print(config)
    inference(out_path=config[0], network_path=config[1], L=6)