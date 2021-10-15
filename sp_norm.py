import numpy as np
from torch import nn

class SP_norm(nn.Module):
    "Specialized batch norm that normalizes per superpixel"

    def __init__(self, in_size, out_size):

        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x, sp):

        return x