import torch.nn as nn
import torch.nn.functional as F
import torch
from ResNetEncoder import ResNetEncoder
from pspnet_parts import *

class PSPNet(nn.Module):

    def __init__(self, in_channels, channel_list, num_classes):
        super(PSPNet, self).__init__()

        self.encoder = ResNetEnoder(in_channels, channel_list)
        self.decoder = PSPNetDecoder(channel_list[-1])
        self.segmentation_head = SegmentationHead(channel_list[-1])

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



