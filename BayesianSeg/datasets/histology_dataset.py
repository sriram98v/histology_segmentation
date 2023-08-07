from collections import namedtuple
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from typing import Callable, Optional, List
import os

class histologyDataset(VisionDataset):
    histologyClass = namedtuple(
        "histology_class",
        ["name", "id", "color"],
    )
    classes = [
        histologyClass("Alveolar_Septa", 0, (111, 74, 0)),
        histologyClass("Cartilage", 1, (81, 0, 81)),
        histologyClass("Red_Blood_Cells", 2, (128, 64, 128)),
        histologyClass("Respiratory_Epithelium", 3, (244, 35, 232)),
        histologyClass("Smooth_Muscle", 4, (250, 170, 160)),
    ]
    def __init__(self, root: str, 
                split: str = "train",
                color: bool = True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                transforms: Optional[Callable] = None,
                classes: Optional[List[str]] = None,
                im_names: Optional[List[str]] = None,
                ):
        super().__init__(root, transforms, transform, target_transform)
        self.split = split
        self.imgs_dir = os.path.join(self.root, self.split, "images")
        self.masks_dir = os.path.join(self.root, self.split, "GT")
        self.color = color
        self.transforms = transforms
        if classes:
            self.classes = classes
        else:    
            self.classes = os.listdir(self.masks_dir)
        self.num_classes = len(self.classes)
        if im_names == None:
            self.im_names = os.listdir(self.imgs_dir)
        else:
            self.im_names = im_names

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        im_name = self.im_names[idx]
        gt_names = [os.path.join(self.masks_dir, gt_class, im_name) for gt_class in self.classes]
        
        if self.color:
            image = F.pil_to_tensor(Image.open(os.path.join(self.imgs_dir, im_name)).convert("RGB"))
        else:
            image = F.pil_to_tensor(Image.open(im_name).convert("L"))
        target = torch.squeeze(torch.stack([F.pil_to_tensor(Image.open(i).convert("L")) for i in gt_names]))/255
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target