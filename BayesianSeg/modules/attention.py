import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F

from .conv import BayesConv1d

class SelfAttention(Module):
    def __init__(self, prior_mu, prior_sigma, n_channels):
        super().__init__()
        self.query,self.key,self.value = [self._conv(prior_mu, prior_sigma, n_channels, c) for c in [n_channels//8,n_channels//8,n_channels]]
        self.gamma = Parameter(torch.tensor([0.]))

    def _conv(self,prior_mu, prior_sigma, n_in,n_out):
        return torch.nn.Conv1d(n_in,n_out, kernel_size=1, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        attn = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, attn) + x
        return o.view(*size).contiguous()