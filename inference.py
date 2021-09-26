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

def inference(source_path='./test_images/source', out_path='./test_images/out/LPTT_L5', network_path='./experiments/LPTT_FiveK_L5_PEG0k3/models/net_g_latest.pth'):
    os.makedirs(out_path, exist_ok=True)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    net = LPTT(num_high=6).to(device)
    load_net = torch.load(network_path)
    net.load_state_dict(load_net, strict=False)
    
    images = glob.glob(source_path+'/*.jpg')
    
    net.eval()
    for imgpath in images:
        imgname = os.path.basename(imgpath)
        with Image.open(imgpath) as img:
            img = img.convert('RGB')
            print(imgname, img.size)
            if img.width*img.height>=6628*8838:
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


if __name__=='__main__':
    inference(out_path='./test_images/out/LPTT_L7', 
              network_path='experiments/LPTT_FiveK_L7/models/net_g_latest.pth')

'''
'./test_images/out/LPTT_L5'
'experiments/LPTT_FiveK_L5_PEG0k3/models/net_g_latest.pth'
'./test_images/out/LPTN_L3'
'./experiments/pretrained_models/net_g_FiveK_numhigh3.pth'
'./test_images/out/LPTT_L6'
'experiments/LPTT_FiveK_L6/models/net_g_latest.pth'
'./test_images/out/LPTT_L7'
'experiments/LPTT_FiveK_L7/models/net_g_latest.pth'
'''