import math
import torch.nn as nn
from misc import ModuleWrapper
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from metrics import calculate_kl as KL_DIV

class BaseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, priors=None, mode="train"):
        super(BaseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda")
        self.mode=mode

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
                'epsilon': 1e-16,
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.epsilon = priors['epsilon']

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.posterior_mu_initial = (-stdv, stdv)
        self.posterior_rho_initial = (-stdv, stdv)
        self.W_mu.data.uniform_(-stdv, stdv)
        self.W_rho.data.uniform_(-stdv, stdv)
        if self.use_bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_rho.data.uniform_(-stdv, stdv)

    def train_mu(self):
        self.W_rho.requires_grad = False
        self.W_mu.requires_grad = True
        self.bias_mu.requires_grad = True
        self.bias_rho.requires_grad = False
    
    def train_rho(self):
        self.W_rho.requires_grad = True
        self.W_mu.requires_grad = False
        self.bias_mu.requires_grad = False
        self.bias_rho.requires_grad = True

    def forward(self, x):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))

        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.conv2d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        
        if self.mode=="train":
            act_std = torch.sqrt(self.epsilon + F.conv2d(x ** 2, self.W_sigma ** 2, bias_var, self.stride, self.padding, self.dilation, self.groups))
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        
        elif self.mode=="infer":
            return act_mu

    def switch_mode(self, mode):
        self.mode=mode

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
        
class CustomDropout(nn.Module):

    """Custom Dropout module to be used as a baseline for MC Dropout"""

    def __init__(self, p:float, activate=True):
        super().__init__()
        self.activate = activate
        self.p = p
    def forward(self, x):
        return nn.functional.dropout(x, self.p, training=self.training or self.activate)
    def switch_activate(self, activate):
        self.activate = activate
    def extra_repr(self):
        return f"p={self.p}, activate={self.activate}"

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, p=0.3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            BaseConv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            BaseConv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # CustomDropout(p=p)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


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
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = BaseConv2d(in_channels, out_channels, kernel_size=1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        return self.out(self.conv(x))


class Bayesian_UNet(ModuleWrapper):
    def __str__(self):
        return "BaysianUNet"

    def __init__(self, n_channels, n_classes, bilinear=True, classes=None):
        super(Bayesian_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.classes = classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def toCPU(self):
        for param in self.parameters():
            param.device("cpu")