
from typing import Tuple
from torchtyping import TensorType

import torch, torch.nn as nn

from ..mappings import normEmbedding, positionalEmbedding
from ..masking import padMask, causalMask
from ..attention import multiHeadedAttention
from ..feedfowardModels.multilayerPerceptron import FFNN
from ..blocks import alignAttention
from ..repeat import stack
from ..outputLayer import logSoftmax


class AttentionClassification(nn.Module):
    """
    An output layer for the transformer layer.
    For the classification task

    Input:
        <encRepr> sequence representation of the input
        <enc_len> length of each batch
    Output:
        <prediction> A probabilistic distribution of each class. 
    """
    def __init__(self, output_size:int, embed_size:int, ff_size:int, nLayers:int, nHeads:int, pDropout:float):
        super().__init__()
        self.startToken  = 0
        self.embTgt      = normEmbedding(1, embed_size)
        self.padMask     = padMask()

        decAttn          = multiHeadedAttention(nHeads, embed_size, pDropout)
        # decAttn_2        = multiHeadedAttention(nHeads, embed_size, pDropout)
        decFF            = FFNN(embed_size, ff_size)
        decLayer         = alignAttention(embed_size, decAttn, decFF, pDropout)
        # self.decoder     = stack(nLayers, decLayer)
        self.decoder     = stack(1, decLayer)
        self.outputLayer = logSoftmax(embed_size, output_size)

    def forward(self,
                    encRepr:    TensorType['batch_size','src_len'],
                    enc_len:    TensorType['batch_size'],
                     )    ->    Tuple[TensorType['batch_size','tgt_len','output_size'],
                                      TensorType['batch_size']]:
        
        x = torch.full_like(enc_len, self.startToken)
        x.unsqueeze_(1) #seq_len = 1
        x_len = torch.full_like(enc_len, 1)
        embTgt    = self.embTgt(x)
        alignMask = self.padMask(x_len, enc_len) #src
        
        out = self.decoder(embTgt, encRepr, alignMask)

        prediction = self.outputLayer(out)
        return prediction