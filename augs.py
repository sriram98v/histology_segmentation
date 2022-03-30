import cv2
import numpy as np
import torch
import torch.nn as nn
import random
from misc import *


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        return {'image': torch.from_numpy(image.copy()).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask.copy()).type(torch.FloatTensor)}

class Resize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None):
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def __call__(self, sample):
        image, mask = torch.unsqueeze(sample['image'], 0), torch.unsqueeze(sample['mask'], 0)

        return {'image': torch.squeeze(nn.functional.interpolate(image, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)),
                'mask': (torch.squeeze(nn.functional.interpolate(mask, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor))>0.5).to(torch.int)
                }

class Rotate(object):
    """Randomly rotate a batch by some multiple of 90 degrees"""
    
    def __call__(self, sample):

        im = sample['image']
        mask = sample['mask']

        deg = random.choice([0, 1, 2, 3])
        
        return {'image':np.rot90(im,k=deg, axes=(1, 2)),
                'mask':np.rot90(mask,k=deg, axes=(1, 2))
                }
        
class Brightness(object):
    """Randomly rotate a batch by some multiple of 90 degrees"""

    def __init__(self, max_brightness=100):
        self.max_brightness = max_brightness
    
    def __call__(self, sample):

        im = np.moveaxis(np.ceil(sample['image']*255).astype(np.uint8), 0, -1)
        mask = sample['mask']

        deg = random.choice(list(range(0, self.max_brightness, 10)))
        
        return {'image':np.moveaxis(increase_brightness(im, value=deg), -1, 0)/255,
                'mask':mask
                }

class RGB_to_HSV(object):
    """Randomly rotate a batch by some multiple of 90 degrees"""
    
    def __call__(self, sample):

        im = np.moveaxis(np.ceil(sample['image']*255).astype(np.uint8), 0, -1)
        mask = sample['mask']

        