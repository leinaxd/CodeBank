from ..normalization import layerNorm
import torch.nn as nn

class selfAttention(nn.Module):
    def __init__(self, embed_size, attn, ff, pDropout):
        super().__init__()
        self.self_attn = attn
        self.feed_forward = ff
        self.dropout = nn.Dropout(pDropout)
        self.norm_1 = layerNorm(embed_size)
        self.norm_2 = layerNorm(embed_size)
        self.size = embed_size
    def forward(self, seq, srcMask):
        """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
        """
        #Block 1
        out = self.norm_1(seq)
        out = self.self_attn(out, out, out, srcMask)
        out = self.dropout(out)
        out = seq + out

        #Block 2
        out = self.norm_2(out)
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = seq + out
        return out