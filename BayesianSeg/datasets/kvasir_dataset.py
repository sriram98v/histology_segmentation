import numpy as np
from torch.utils.data import Dataset
import os
import cv2


class kvasirDataset(Dataset):
    def __init__(self, imgs_dir, gt_dir, classes=None, transform=None, color=False, im_names=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = gt_dir
        if classes:
            self.classes = classes
        else:    
            self.classes = os.listdir(gt_dir)
        self.num_classes = len(self.classes)
        self.classes.sort()
        if im_names == None:
            self.im_names = os.listdir(self.imgs_dir)
        else:
            self.im_names = im_names
        self.transform = transform
        self.color = color

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        im_name = self.im_names[idx]
        gt_names = [os.path.join(self.masks_dir, gt_class, im_name) for gt_class in self.classes]
        
        if self.color:
            img = cv2.imread(os.path.join(self.imgs_dir, im_name))
            img = np.moveaxis(img, -1, 0)
        else:
            img = cv2.imread(os.path.join(self.imgs_dir, im_name), 0)
            img = np.expand_dims(img, axis=0)
        mask = np.array([cv2.imread(i, 0) for i in gt_names])

        image, target = img/255, mask

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
    
    def append_ims(self, new_ims):
        self.im_names += new_ims

    def remove_ims(self, im_list):
        for i in im_list:
            self.im_names.remove(i[0])

    def im_name(self, idx):
        return self.im_names[idx]

