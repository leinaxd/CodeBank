from ..normalization import layerNorm
import torch.nn as nn

import copy

class stack(nn.Module):
    """Produce N identical layers."""
    def __init__(self, nStacks, layer):
        super().__init__()
        self.nStacks = nStacks
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(nStacks)])
        self.norm = layerNorm(layer.size)       
    def forward(self, x, *args, **kwargs):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers: x = layer(x, *args, **kwargs)
        return self.norm(x)
        
#TODO
class stack_2(nn.Module):
    """Produce N identical layers."""
    def __init__(self, nStacks, layer):
        super().__init__()
        self.nStacks = nStacks
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(nStacks)])
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers: 
            x = layer(x, mask)
        return x