import torch.nn as nn
import torch
import torch.nn.functional as F
from torchbnn import nn as bnn

class DoubleConv(nn.Module):
    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            bnn.BayesConv2d(prior_mu, prior_sigma, in_channels, mid_channels, kernel_size=3, padding=1),
            bnn.BayesBatchNorm2d(prior_mu, prior_sigma, mid_channels),
            nn.ReLU(),
            bnn.BayesConv2d(prior_mu, prior_sigma, mid_channels, out_channels, kernel_size=3, padding=1),
            bnn.BayesBatchNorm2d(prior_mu, prior_sigma, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(prior_mu, prior_sigma, in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(prior_mu, prior_sigma, in_channels, out_channels, in_channels // 2)
        else:
            self.up = bnn.BayesConvTransposed2d(prior_mu, prior_sigma, in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(prior_mu, prior_sigma, in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = bnn.BayesConv2d(prior_mu, prior_sigma, in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        output = self.conv(x)
        return output

class Bayesian_UNet(bnn.BayesModule):
    def __str__(self):
        return "BaysianUNet"

    def __init__(self, prior_mu, prior_sigma, n_channels, n_classes, bilinear=True, classes=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.classes = classes

        self.inc = DoubleConv(prior_mu, prior_sigma, n_channels, 64)
        self.down1 = Down(prior_mu, prior_sigma, 64, 128)
        self.down2 = Down(prior_mu, prior_sigma, 128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(prior_mu, prior_sigma, 256, 512 // factor)
        self.up1 = Up(prior_mu, prior_sigma, 512, 256 // factor, bilinear)
        self.up2 = Up(prior_mu, prior_sigma, 256, 128 // factor, bilinear)
        self.up3 = Up(prior_mu, prior_sigma, 128, 64, bilinear)
        self.out = OutConv(prior_mu, prior_sigma, 64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        outs = self.out(x)
        return outs
    
# class Bayesian_UNet(nn.Module):
#     def __str__(self):
#         return "BaysianUNet"

#     def __init__(self, prior_mu, prior_sigma, n_channels, n_classes, bilinear=True, classes=None, out_layer="sigmoid", crf_out=True, n_iter=5):
#         super(Bayesian_UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.classes = classes
#         self.n_iter = n_iter

#         self.backbone = Bayesian_UNet_backbone(prior_mu, prior_sigma, n_channels, n_classes, bilinear=True, classes=None, out_layer=out_layer)
#         if crf_out:
#             self.crf = modules.CRF(self.n_iter, self.n_channels, n_classes, kernel_size=[3, 3],
#                     theta_alpha=[1.5, 2.5], theta_beta=[1.5, 2.5], theta_gamma=[1.5,])
#         self.crf_out = crf_out

#     def freeze_crf(self, x=True):
#         if self.crf_out:
#             self.crf.requires_grad_(not x)

#     def freeze_backbone(self, x=True):
#         self.backbone.requires_grad_(not x)
    
#     def train_backbone(self):
#         self.freeze_crf(True)
#         self.freeze_backbone(False)
#         self.crf_out = False

#     def train_crf(self):
#         if self.crf_out:
#             self.freeze_backbone(True)
#             self.freeze_crf(False)
#             self.crf_out = True

#     def train(self):
#         self.freeze_backbone(False)
#         self.freeze_crf(False)

#     def eval(self):
#         self.freeze_backbone(True)
#         self.freeze_crf(True)

#     def forward(self, x):
#         probs = self.backbone(x)
#         if self.crf_out:
#             probs = self.crf(probs, x)

#         return probs

        