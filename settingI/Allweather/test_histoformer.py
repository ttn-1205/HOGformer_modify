import cv2
import kornia
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
from Allweather import util

from natsort import natsorted
from glob import glob
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from basicsr.models.archs.hogformer_arch import HOGformer
from skimage import img_as_ubyte
from pdb import set_trace as stx
import time

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default='', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./RainDrop/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./Allweather/pretrained_models/net_g_latest.pth', type=str, help='Path to weights')
parser.add_argument('--yaml_file', default='./Allweather/Options/Allweather_HOGformer.yml', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = args.yaml_file
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = HOGformer(**x['network_g'])

checkpoint = torch.load(args.weights)
'''
from thop import profile
flops, params = profile(model_restoration, inputs=(torch.randn(1, 3, 256,256), ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
'''
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration = model_restoration.cuda()
model_restoration.eval()
factor = 16
result_dir  = os.path.join(args.result_dir)
os.makedirs(result_dir, exist_ok=True)
inp_dir = os.path.join(args.input_dir)
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))

# Add function for tiling --> prevent OOM
def safe_forward(model, input_, tile_size=256, overlap=32):
    b, c, h, w = input_.shape
    stride = tile_size - overlap

    device = input_.device
    output = torch.zeros((b, c, h, w), device=device)
    weight = torch.zeros((b, c, h, w), device=device)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)

            tile = input_[:, :, y1:y2, x1:x2]

            # padding (same logic as original, just local)
            pad_h = tile_size - (y2 - y1)
            pad_w = tile_size - (x2 - x1)
            if pad_h > 0 or pad_w > 0:
                if (tile.shape[2] <= pad_h) or (tile.shape[3] <= pad_w):
                    # fallback for very small tiles
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='replicate')
                else:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            out_tile = model(tile)

            out_tile = out_tile[:, :, :y2-y1, :x2-x1]

            output[:, :, y1:y2, x1:x2] += out_tile
            weight[:, :, y1:y2, x1:x2] += 1.0

    return output / weight

with torch.no_grad():
    for file_ in tqdm(files):
        # Remove from original code
        # torch.cuda.ipc_collect()
        # torch.cuda.empty_cache()
        img = np.float32(util.load_img(file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()
        # Padding in case images are not multiples of factor
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        time1 = time.time()

        # New code compared to old restored line
        restored = safe_forward(model_restoration, input_)
        # restored = model_restoration(input_)
        # hybrid degradation
        # restored = model_restoration(restored)
        time2 = time.time()
        print(time2-time1)
        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        util.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
