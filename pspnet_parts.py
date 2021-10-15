import torch.nn as nn
import torch.nn.functional as F
import torch

class EncoderHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

    def forward(self, x):
        return self.encoder_head(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class PoolBlock(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_block = nn.Sequential(
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x):

        return self.pool_block(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, scale_factor, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)


    def forward(self, x):
        return self.up(x)

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        # self.up = nn.Upsample(4, mode='bilinear', align_corners=True)
        self.out = nn.Sigmoid()

    def forward(self, x):
        return self.out(self.conv1(x))

class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels):

        self.conv_block1 = ConvBlock(in_channels, out_channels)
        self.conv_block2 = ConvBlock(out_channels, out_channels)
        self.conv_block3 = ConvBlock(out_channels, out_channels)
        self.conv_block4 = ConvBlock(out_channels, out_channels)


# class ResnetEncoder(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()

#         self.

