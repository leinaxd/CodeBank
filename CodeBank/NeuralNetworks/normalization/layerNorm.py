from torchtyping import TensorType
from typing import Any
import torch, torch.nn as nn

class layerNorm(nn.Module):
    """Construct a layernorm module (See citation for details).
    layerNorm({x1 x2 .. xn}) = 
    """
    def __init__(self, features_size, eps=1e-6):
        super().__init__()
        self.fixedStd  = nn.Parameter(torch.ones(features_size))
        self.fixedMean = nn.Parameter(torch.zeros(features_size))
        self.eps = eps
    def forward(self, x:TensorType['batch_size', Any]):
        layerMean = x.mean(-1, keepdim=True)
        layerStd = x.std(-1, keepdim=True)
        return self.fixedStd * (x - layerMean) / (layerStd + self.eps) + self.fixedMean