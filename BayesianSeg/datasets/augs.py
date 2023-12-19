from typing import Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import random
from BayesianSeg.misc import *

class PIL_to_tensor(object):
    def __call__(self, img, target):
        """Convert a ``PIL Image`` to a tensor of the same type.
        This function does not support torchscript.
    
        See :class:`~torchvision.transforms.PILToTensor` for more details.
    
        .. note::
    
            A deep copy of the underlying array is performed.
    
        Args:
            pic (PIL Image): Image to be converted to tensor.
    
        Returns:
            Tensor: Converted image.
        """
        # handle PIL Image
        img = torch.as_tensor(np.array(img, copy=True))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1))

        
        return img, torch.as_tensor(np.array(target, copy=True))

class Resize(object):
    """Resize tensor"""

    def __init__(self, size=None, max_size=None, antialias=True, interpolation=F.InterpolationMode.NEAREST):
        self.size = size
        self.max_size = max_size
        self.antialias = antialias
        self.interpolation = interpolation

    def __call__(self, image, target):
        # target = torch.unsqueeze(torch.unsqueeze(target, dim=0), dim=0)

        return F.resize(image, self.size, self.interpolation, self.max_size, self.antialias), F.resize(target, self.size, self.interpolation, self.max_size, self.antialias)

class Rotate(object):
    """Randomly rotate a batch by some multiple of 90 degrees"""

    def __init__(self, degree, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=0):

        self.degree = degree

        self.center = center

        self.interpolation = interpolation
        self.expand = expand

        self.fill = fill

    @staticmethod
    def get_params(degree):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(-float(degree), float(degree)).item())
        return angle
    
    def __call__(self, image, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        channels_im, _, _ = F.get_dimensions(image)
        channels_tar, _, _ = F.get_dimensions(target)
        fill_im = [float(fill)] * channels_im
        fill_target = [float(0)] * channels_tar
        angle = self.get_params(self.degree)

        return F.rotate(image, angle, self.interpolation, self.expand, self.center, fill_im), F.rotate(target, angle, self.interpolation, self.expand, self.center, fill_target)

        
class Brightness(object):

    def __init__(self, max_brightness=100):
        self.max_brightness = max_brightness
    
    def __call__(self, image, target):

        im = np.moveaxis(np.ceil(image*255).astype(np.uint8), 0, -1)

        deg = random.choice(list(range(0, self.max_brightness, 10)))
        
        return np.moveaxis(increase_brightness(im, value=deg), -1, 0)/255, target
    
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
        new_target = torch.zeros((self.num_classes, target.shape[0], target.shape[1]))

        for i in range(1, self.num_classes):
            new_target[i] = (target==i)

        return image, new_target
    
class norm_im(object):
    def __call__(self, image, target):
        return image/255, target
    
class gauss_noise(object):
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0)):
        self.kernel_size = (kernel_size, kernel_size)
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma]), target
