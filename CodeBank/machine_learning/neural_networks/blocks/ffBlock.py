import torch.nn as nn
from ..normalization import layerNorm

class ffBlock(nn.Module):
    """
    ffBlock
        Applies a residual connection and normalization to a feedfoward module


    """
    def __init__(self, embed_size, ff, pDropout):
        super().__init__()
        self.size = embed_size
        self.feed_forward = ff
        self.dropout    = nn.Dropout(pDropout)
        self.norm       = layerNorm(embed_size)

    def forward(self, x):
        """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
        """

        #feed foward
        out = self.norm(x)
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = x + out

        return out