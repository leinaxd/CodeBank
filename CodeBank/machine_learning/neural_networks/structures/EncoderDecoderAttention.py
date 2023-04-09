from ..torchModule import torchModule
import torch, torch.nn as nn
from torchtyping import TensorType
from typing import Tuple, List


class EncoderDecoderAttention(nn.Module):
    """Transforms an input sequence into an output sequence"""
    def __init__(self, encoderModel, decoderModel, attentionModel, projEncDec=None):
        super().__init__()
        self.encoderModel = encoderModel
        self.decoderModel = decoderModel
        self.attentionModel = attentionModel
        self.dec_init_proj  = projEncDec

    def forward(self, seq:      TensorType['batch_size','seq_len',int],
                      seq_len:  TensorType['batch_size',int], 
                      )->       TensorType['batch_size','tgt_len','vocab_len',int]:
        enc_hiddens, enc_final_state = self.encoderModel(seq, seq_len)

        dec_init_state = (self.dec_init_proj(enc_final_state[0]), enc_final_state[1])
        out, out_len = self.decoderModel(dec_init_state)
        # embedPredictions = torch.stack(predictions) #(tgt_len, batch_size, dec_hidden_size)
        # vocabPredictions = self.target_vocab_projection(embedPredictions) #From embed_size to Vocab_size
        # logP = F.log_softmax(vocabPredictions, dim=-1)
        return out, out_len

#TODO:
#   -Refactorizar en secuential