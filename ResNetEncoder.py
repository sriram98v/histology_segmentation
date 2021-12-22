import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip=False):
        super().__init__()
        self.skip = skip
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if skip:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
                                      nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

    def forward(self, x):
        if self.skip:
            return self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x)))))+self.skip(x)
        else:
            return self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x)))))

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p0 = BasicBlock(in_channels, out_channels, skip=True)
        self.p1 = BasicBlock(out_channels, out_channels, skip=False)
        self.p2 = BasicBlock(out_channels, out_channels, skip=False)
        self.p3 = BasicBlock(out_channels, out_channels, skip=False)
        
    def forward(self, x):
        return self.p3(self.p2(self.p1(self.p0(x))))


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.norm = nn.BatchNorm2d(channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer0 = ResNetLayer(channels[0], channels[0])
        self.layer1 = ResNetLayer(channels[0], channels[1])
        self.layer2 = ResNetLayer(channels[1], channels[2])
        self.layer3 = ResNetLayer(channels[2], channels[3])
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.norm(self.conv(x))))
        return self.layer3(self.layer2(self.layer1(self.layer0(x))))