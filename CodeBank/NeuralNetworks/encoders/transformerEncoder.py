from torchtyping import TensorType

import torch.nn as nn

from ..mappings import normEmbedding, positionalEmbedding
from ..masking import padMask
from ..attention import multiHeadedAttention
from ..feedfowardModels.multilayerPerceptron import FFNN
from ..repeat import stack

from ..blocks import selfAttention
#TODO:
#   Refactorizar el standard TransformerModel.encoderLayer Aqui mismo

class transformerEncoder(nn.Module):
    """
    Encoder Module of Transformed as seen in
    Paper: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, vocab_size, embed_size, ff_size, nLayers, nHeads, pDropout):
        super().__init__()
        self.padMask    = padMask()
        self.posEmb     = positionalEmbedding(embed_size)
        self.normEmbed  = normEmbedding(vocab_size, embed_size)
        self.src_embed  = nn.Sequential(self.normEmbed, self.posEmb)

        encAttn         = multiHeadedAttention(nHeads, embed_size, pDropout)
        encFF           = FFNN(embed_size, ff_size)
        encLayer        = selfAttention(embed_size, encAttn, encFF, pDropout)
        self.encoder    = stack(nLayers, encLayer)
    def forward(self,
                    src:        TensorType['batch_size','src_len',int],
                    src_len:    TensorType['batch_size',int],
                     )    ->    TensorType['batch_size','src_len','embed_size']:
        embSrc       = self.src_embed(src)
        srcMask      = self.padMask(src_len, src_len)
        srcRepr      = self.encoder(embSrc, srcMask)
        return srcRepr
