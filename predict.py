import numpy as np
import torch
from Fpn import FPN
import cv2
import matplotlib.pyplot as plt
import math
import seaborn as sns

def predict_image(im, model, device , window_size = 512, step_size = 256):
    im_h, im_w = im.shape
    out = np.zeros((9, im_h, im_w))
    
    batch = []
    for i in range(0, im_h-window_size, step_size):
        for j in range(0, im_w-window_size, step_size):
            batch.append(im[i:i+window_size, j:j+window_size]/255)
        batch.append(im[i:i+window_size, im_w-window_size:im_w]/255)
    batch.append(im[im_h-window_size:im_h, im_w-window_size:im_w]/255)
    batch = torch.Tensor(np.expand_dims(np.array(batch), axis=1)).to(device=device)
    outs = model(batch).detach().cpu().numpy()
    n = 0
    for i in range(0, im_h-window_size, 256):
        for j in range(0, im_w-window_size, 256):
            out[:, i:i+window_size, j:j+window_size] = outs[n]
            n += 1
        out[:, i:i+window_size, im_w-window_size:im_w] = outs[n]
        n +=1
    out[:, im_h-window_size:im_h, im_w-window_size:im_w] = outs[n]
    
    return out