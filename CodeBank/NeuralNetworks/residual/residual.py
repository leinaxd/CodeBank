import torch.nn as nn
from ..normalization import layerNorm

class residualConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        out = self.module(x)
        return x + out