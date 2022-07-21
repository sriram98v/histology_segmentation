import numpy as np
import torch
from BayesianSeg.models.bayes_unet import *
import cv2
import matplotlib.pyplot as plt
import math
import os
import torch.nn as nn
from skimage.segmentation import slic
from BayesianSeg.misc import *
from tqdm import tqdm
import argparse
import sys
import json

def init_model(args):
    try:
        with open(args.classes, 'r') as f:
            classes = json.load(f)
        model = Bayesian_UNet(3, 5, classes=classes)
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        return model
    except:
        return False

def get_args():
    parser = argparse.ArgumentParser(description='Segments histology images in input directory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-dir', type=str, help='Input Directory', dest='in_dir', default="./input/")
    parser.add_argument('-o', '--out-dir', type=str, default="./output/", help='Output Directory', dest='out_dir')
    parser.add_argument('-m', '--model', type=str, default="./model.pth", help='Pretrained model', dest='model')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='device', dest='device')
    parser.add_argument('-c', '--classes', type=str, default='None', help='Classes file', dest='classes')


    

    return parser.parse_args()


def output_im(im, window_size = 512, step_size = 256, size=(256, 256)):
    im_h, im_w = im.shape[:-1]
    batch = []
    for i in range(0, im_h-window_size, step_size):
        for j in range(0, im_w-window_size, step_size):
            batch.append(im[i:i+window_size, j:j+window_size])
        batch.append(im[i:i+window_size, im_w-window_size:im_w])
    for j in range(0, im_w-window_size, step_size):
        batch.append(im[im_h-window_size:im_h, j:j+window_size])
    batch.append(im[im_h-window_size:im_h, im_w-window_size:im_w])
    
    batch = np.moveaxis(np.array(batch), -1, 1)
    batch = nn.functional.interpolate(torch.from_numpy(batch)/255, size=size, 
                                     scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None).to(device=torch.device(args.device if torch.cuda.is_available() else 'cpu'))
#     print(batch.shape)
    if args.device=="cuda":
        outs = model(batch).detach()
    else:
        outs = model(batch)
    
    outs = nn.functional.interpolate(outs, size=(512, 512), 
                                     scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None).detach().cpu().numpy()
    out = np.zeros((5, im_h, im_w))
    n = 0
    for i in range(0, im_h-window_size, 256):
        for j in range(0, im_w-window_size, 256):
            out[:, i:i+window_size, j:j+window_size] = outs[n]
            n += 1
        out[:, i:i+window_size, im_w-window_size:im_w] = outs[n]
        n +=1
    for j in range(0, im_w-window_size, 256):
        out[:, im_h-window_size:im_h, j:j+window_size] = outs[n]
        n += 1
    out[:, im_h-window_size:im_h, im_w-window_size:im_w] = outs[n]
    
#     out_2 = np.zeros_like(out)
    
#     segments = slic(im, n_segments=1000, compactness=28)
#     segments = segments
    
#     for j in range(out_2.shape[0]):
#         for i in np.unique(segments):
#             out_2[j][np.where(segments==i)] = np.mean(out[j][np.where(segments==i)], axis=0)
    
    return out

def highlight_im(im, masks, threshold=0.5):
    threshold = int(threshold*255)
    mask = (masks*255).astype(np.uint8)
    ret, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(im, contour, -1, (255,0,0), thickness = 1)

    b,g,r = cv2.split(im)

    b = cv2.add(b, 100, dst = b, mask = binary, dtype = cv2.CV_8U)
    g = cv2.add(g, 100, dst = g, mask = binary, dtype = cv2.CV_8U)
#     r = cv2.add(r, 100, dst = r, mask = binary, dtype = cv2.CV_8U)

    return cv2.merge((b,g,r), im)


def entropy(probs):
#     log_prob = np.log(out)
    return -np.sum(probs*np.nan_to_num(np.log2(probs)), axis=0)

def entropy_sampling(im, max_iter=20):
    model.switch_custom_dropout(activate=True)
    out = np.zeros((model.n_classes, im.shape[0], im.shape[1]))
    
    for i in tqdm(range(max_iter)):
        out += output_im(im)
    out = out/max_iter
    
    out_2 = np.zeros_like(out)
    segments = slic(im, n_segments=1000, compactness=28)
    segments = segments
    
    for j in range(out_2.shape[0]):
        for i in np.unique(segments):
            out_2[j][np.where(segments==i)] = np.mean(out[j][np.where(segments==i)], axis=0)
    
    return entropy(out_2)

def make_outs(im, out, args, im_name):
    fig, axs = plt.subplots(3, 3, figsize=(20, 10))
    for i in axs:
        for j in i:
            j.axis('off')
    axs[0, 0].set_title("Input Image")
    axs[0, 0].imshow(im)
    # axs[0, 0].set_title(i, fontsize=50)
    for n,i in enumerate(range(out.shape[0])):
        axs[int(i/3)+1, int(i%3)].imshow(highlight_im(im.copy(), out[i].copy(), threshold=0.5))
        axs[int(i/3)+1, int(i%3)].set_title(model.classes[n])
    
    fig.savefig(os.path.join(args.out_dir, im_name.split("/")[-1])+".png")
    # fig.close()

if __name__=="__main__":
    args = get_args()
    model = init_model(args)
    if model==0:
        sys.exit("Model path incorrect")
        
    input_ims = [os.path.join(args.in_dir, i) for i in os.listdir(args.in_dir)]
    
    for i in tqdm(input_ims):
        im = cv2.imread(i)

        out = output_im(im)
        make_outs(im, out, args, i)
