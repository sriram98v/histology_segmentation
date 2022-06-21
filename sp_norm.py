import numpy as np
from torch import nn
import torch

class SP_norm(nn.Module):
    "Specialized batch norm that normalizes per superpixel"

    def __init__(self):
        pass

    def forward(self, x, sp):

        x_mean = torch.mean(x, dim=0)
        x_std = torch.var(x, dim=0, unbiased=False)

        out = torch.zeros_like(x)
        for channel in x.shape[0]:
            for sp_id in torch.unique(sp):
                out[channel][torch.argwhere(x==sp_id)] = torch.mean(x[torch.argwhere(x==sp_id)])
        
        return out