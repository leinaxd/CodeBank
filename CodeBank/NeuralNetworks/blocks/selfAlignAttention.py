import torch.nn as nn
from ..normalization import layerNorm

class selfAlignAttention(nn.Module):
    """
    selfAlignAttention block
        Aligns a sequence with another sequence

    Method
        Apply a self-attention block, following an align attention and finally a feedfoward network

    """
    def __init__(self, embed_size, attn_1, attn_2, ff, pDropout):
        super().__init__()
        self.size = embed_size
        self.self_attn = attn_1
        self.src_attn = attn_2
        self.feed_forward = ff
        self.dropout = nn.Dropout(pDropout)
        self.norm_1 = layerNorm(embed_size)
        self.norm_2 = layerNorm(embed_size)
        self.norm_3 = layerNorm(embed_size)

    def forward(self, x, memory, align_mask, tgt_mask):
        """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
        """
        m = memory

        #self-attention
        out = self.norm_1(x)
        out = self.self_attn(out, out, out, tgt_mask)
        out = self.dropout(out)
        out = x + out

        #alignament-attention
        x=out
        out = self.norm_2(x)
        out = self.src_attn(out, m, m, align_mask)
        out = self.dropout(out)
        out = x + out

        #feed foward
        x=out
        out = self.norm_3(x)
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = x + out

        return out