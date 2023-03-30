
from typing import Tuple
from torchtyping import TensorType

import torch.nn as nn

from ..mappings import normEmbedding, positionalEmbedding
from ..masking import padMask, causalMask
from ..attention import multiHeadedAttention
from ..feedfowardModels.multilayerPerceptron import FFNN
from ..blocks.selfAlignAttention import selfAlignAttention
from ..repeat import stack
from ..outputLayer import logSoftmax

class transformerDecoder(nn.Module):
    """
    Decoder Module of Transformed as seen in
    Paper: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, dec_input_size, embed_size, output_size, ff_size, nLayers, nHeads, pDropout):
        super().__init__()
        dec_posEmb       = positionalEmbedding(embed_size)
        tgt_normEmbed    = normEmbedding(dec_input_size, embed_size)
        self.tgt_embed   = nn.Sequential(tgt_normEmbed, dec_posEmb)

        self.padMask     = padMask()
        self.causalMask  = causalMask()

        decAttn_1        = multiHeadedAttention(nHeads, embed_size, pDropout)
        decAttn_2        = multiHeadedAttention(nHeads, embed_size, pDropout)
        decFF            = FFNN(embed_size, ff_size)
        decoderLayer     = selfAlignAttention(embed_size, decAttn_1, decAttn_2, decFF, pDropout)
        self.decoder     = stack(nLayers, decoderLayer)
        self.outputLayer = logSoftmax(embed_size, output_size)
        
    def forward(self,
                    seq:        TensorType['batch_size','tgt_len',int],
                    seq_len:    TensorType['batch_size',int],
                    encRepr:    TensorType['batch_size','src_len'],
                    enc_len:    TensorType['batch_size'],
                    # alignMask:  TensorType['batch_size','tgt_len','src_len'],
                    # tgtMask:    TensorType['batch_size','tgt_len','tgt_len'],
                     )    ->    Tuple[TensorType['batch_size','tgt_len','output_size'],
                                      TensorType['batch_size']]:
        embTgt    = self.tgt_embed(seq)

        tgtMask   = self.padMask(seq_len, seq_len)
        tgtMask  &= self.causalMask(seq)
        alignMask = self.padMask(seq_len, enc_len)

        out = self.decoder(embTgt, encRepr, alignMask, tgtMask)

        prediction = self.outputLayer(out)
        return prediction