from .histology_dataset import *
from .glas_dataset import *
from .augs import *
import torchvision

def DS(name, root_dir, split, transforms, color=True, classes = None, im_names = None):
    match name:
        case "cityscapes":
            return torchvision.datasets.Cityscapes(root_dir, split=split, mode="fine",
                     target_type='semantic', transforms=BSCompose(transforms))
        case "histo":
            return histologyDataset(root_dir, split=split, color=color, transforms=BSCompose(transforms), classes=classes, im_names=im_names)
        case "glas":
            return GlaSDataset(root_dir, split=split, color=color, transforms=BSCompose(transforms), classes=classes, im_names=im_names)
        case _:
            raise NameError(f'Dataset {name} not found')