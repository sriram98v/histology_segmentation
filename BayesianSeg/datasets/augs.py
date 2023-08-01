from typing import Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import random
from BayesianSeg.misc import *


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, target):

        return torch.from_numpy(image.copy()).type(torch.FloatTensor), torch.from_numpy(target.copy()).type(torch.FloatTensor)

class Resize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None):
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def __call__(self, image, target):
        image, mask = torch.unsqueeze(image, 0), torch.unsqueeze(target, 0)

        return torch.squeeze(nn.functional.interpolate(image, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)), (torch.squeeze(nn.functional.interpolate(mask, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor))>0.5).to(torch.int)
            

class Rotate(object):
    """Randomly rotate a batch by some multiple of 90 degrees"""
    
    def __call__(self, image, target):

        im = image
        mask = target

        deg = random.choice([0, 1, 2, 3])
        
        return np.rot90(im,k=deg, axes=(1, 2)), np.rot90(mask,k=deg, axes=(1, 2))

        
class Brightness(object):

    def __init__(self, max_brightness=100):
        self.max_brightness = max_brightness
    
    def __call__(self, image, target):

        im = np.moveaxis(np.ceil(image*255).astype(np.uint8), 0, -1)

        deg = random.choice(list(range(0, self.max_brightness, 10)))
        
        return np.moveaxis(increase_brightness(im, value=deg), -1, 0)/255, target

    
class to_np(object):
    """Converts PIL to np array"""
    
    def __call__(self, image, target):

        return np.array(image), np.array(target)
    
class BSCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class class_to_channel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, image, target):
        new_target = np.zeros((self.num_classes, target.shape[0], target.shape[1]))

        for i in range(1, self.num_classes):
            new_target[i] = (target==i)

        return image, new_target
    
class reorder_im_shape(object):
    def __call__(self, image, target):
        img = np.zeros((image.shape[2], image.shape[0], image.shape[1]))

        for i in range(image.shape[2]):
            img[i] = image[:, :, i]

        return img, target
    
class norm_im(object):
    def __call__(self, image, target):
        return image/255, target
