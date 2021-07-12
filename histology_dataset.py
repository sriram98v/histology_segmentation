import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2


class histologyDataset(Dataset):
    def __init__(self, imgs_dir, gt_dir, classes=None, augs=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = gt_dir
        if classes:
            self.classes = classes
        else:    
            self.classes = os.listdir(gt_dir)
        self.num_classes = len(self.classes)
        self.classes.sort()
        self.im_names = os.listdir(self.imgs_dir)
        self.augs = augs

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        im_name = self.im_names[idx]
        gt_names = [os.path.join(self.masks_dir, gt_class, im_name) for gt_class in self.classes]
        
        img = np.expand_dims(cv2.imread(os.path.join(self.imgs_dir, im_name), 0), axis=0)
        mask = np.array([cv2.imread(i, 0) for i in gt_names])

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }