import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101

class deeplabv3(nn.Module):
    def __init__(self, pretrained=False, progress=True, num_classes=9):
        super(deeplabv3, self).__init__()

        self.head = deeplabv3_resnet101(pretrained=pretrained, progress=progress, num_classes=num_classes)
        
        self.out = nn.Sigmoid()
        
    def forward(self, x):

        x = self.head(x)
        return self.out(x["out"])
