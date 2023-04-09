import torch.nn as nn
from ..normalization import layerNorm

class alignAttention(nn.Module):
    """
    alinAttention block
        Aligns a sequence with a sequence of features

    Method
        Apply a self-attention block, following an align attention and finally a feedfoward network

    """
    def __init__(self, embed_size, attn_1, ff, pDropout):
        super().__init__()
        self.size = embed_size
        self.align_attn = attn_1
        self.feed_forward = ff
        self.dropout = nn.Dropout(pDropout)
        self.norm_1 = layerNorm(embed_size)
        self.norm_2 = layerNorm(embed_size)

    def forward(self, x, memory, align_mask=None):

        m = memory
       
        #alignament-attention
        out = self.norm_1(x)
        out = self.align_attn(out, m, m, align_mask)
        out = self.dropout(out)
        out = x + out

        #feed foward
        x=out
        out = self.norm_2(x)
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = x + out
        return out