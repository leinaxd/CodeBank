from ..IOmodule import seq2seq
from ..structures import Encoder
from ..structures import Decoder
from ..structures import EncoderDecoderAttention

import torch.nn as nn

class rnnModel(seq2seq):
    def __init__(self, debug, input_size, output_size, embed_size, padToken, iniToken, endToken):
        super().__init__(padToken, endToken)
        hidden_size         = 32   if debug else 1024
        dropout_rate        = 0     if debug else 0.3#0.3
        nStacks             = 1     if debug else 2#4
        nDepth              = 1     if debug else 2
        residual            = True  if debug else True
        attType             = 'mult'
        recurrentUnit       = 'resRNN'
        # embed_size  = 2 #Ultra fast
        # hidden_size = 4


        seqEmb = nn.Embedding(input_size, embed_size, padToken)

        enc_unit    = nn.LSTM(embed_size, hidden_size, num_layers=nStacks, batch_first=True, bidirectional=False)
        dec_unit    = nn.LSTM(embed_size, hidden_size, num_layers=nStacks, batch_first=True, bidirectional=False)
        attention   = None
        projectionED  = nn.Linear(hidden_size, hidden_size)
        projectionDO  = nn.Linear(hidden_size, output_size)
        encoder     = Encoder(enc_unit,seqEmb)
        decoder     = Decoder(dec_unit, seqEmb, projectionDO, iniToken, endToken)
        self.EDA    = EncoderDecoderAttention(encoder, decoder, attention, projectionED)
    def forward(self, seq, seq_len):
        return self.EDA.forward(seq, seq_len)