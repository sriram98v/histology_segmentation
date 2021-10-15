import torch.nn as nn
import torch.nn.functional as F
import torch
from pspnet_parts import *

class PSPNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(PSPNet, self).__init__()

        self.encoder = EncoderHead(in_channels)

        self.block0 = ConvBlock(64, 64)
        self.block1 = ConvBlock(64, 64)
        self.block2 = ConvBlock(64, 64)

        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 128)
        self.block5 = ConvBlock(128, 128)
        
        self.block6 = ConvBlock(128, 256)
        self.block7 = ConvBlock(256, 256)
        self.block8 = ConvBlock(256, 256)

        self.block9 = ConvBlock(256, 512)
        self.block10 = ConvBlock(512, 512)
        self.block11 = ConvBlock(512, 512)


        self.pool0 = PoolBlock(2)
        self.pool1 = PoolBlock(4)
        self.pool2 = PoolBlock(8)
        self.pool3 = PoolBlock(16)

        self.postpool0 = ConvBlock(512, 512)
        self.postpool1 = ConvBlock(512, 512)
        self.postpool2 = ConvBlock(512, 512)
        self.postpool3 = ConvBlock(512, 512)

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.final_block0 = ConvBlock(512*5, 512)
        self.final_block1 = ConvBlock(512, 256)
        self.final_block2 = ConvBlock(256, 128)
        self.final_block3 = ConvBlock(128, 64)

        self.up_out = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.out = SegmentationHead(64, num_classes)

    def forward(self, xin):

        x = self.block2(self.block1(self.block0(self.encoder(xin))))
        x = self.block5(self.block4(self.block3(x)))
        x = self.block8(self.block7(self.block6(x)))
        x = self.block11(self.block10(self.block9(x)))

        
        x0 = self.up0(self.postpool0(self.pool0(x)))
        x1 = self.up1(self.postpool1(self.pool1(x)))
        x2 = self.up2(self.postpool2(self.pool2(x)))
        x3 = self.up3(self.postpool3(self.pool3(x)))
        
        x = self.final_block0(torch.cat([x, x0, x1, x2, x3], dim=1))

        x = self.final_block3(self.final_block2(self.final_block1(x)))

        return self.out(self.up_out(x))



