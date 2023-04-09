#Attention libraries
#   Includes:
#       dotAttention
#
#
from typing import Tuple, List
from torchtyping import TensorType
import torch
import torch.nn as nn
import torch.nn.functional as F

class luongAttention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, device):
        #Todo:
        #   key_size, query_size, value_size
        super().__init__()
        #v_t = W_u [a_t; h_t^dec]
        self.combined_output_projection = nn.Linear(enc_hidden_size+dec_hidden_size, dec_hidden_size, False, device=device) 
        self.mask = None
        self.device = device
    def forward(self, 
                    values: TensorType['batch_size', 'src_len', 'enc_hidden_size'], 
                    keys:   TensorType['batch_size', 'src_len', 'dec_hidden_size'], 
                    query:  TensorType['batch_size', 'dec_hidden_size']
                    ) ->    Tuple[Tuple, torch.Tensor, torch.Tensor]: #recibe generadores
        """ Compute one forward step of the LSTM decoder, attention computation."""
        # alpha_i,t = softmax( k*q )
        # c_t = alpha_i,t v_i
        # U_t = [query; a_t]
        # out = tanh(W U_t)

        #Similarity function k*q
        e_t = torch.bmm(keys, query.unsqueeze(-1)).squeeze(2)  #(b x src_len x h) · (b x h x 1) = (b x src_len x 1)

        # Set e_t to -inf where enc_masks has 1
        if self.mask is not None:
            e_t.data.masked_fill_(self.mask.bool(), float('-inf'))

        alpha_t = F.softmax(e_t, dim = 1)
        c_t = torch.bmm(alpha_t.unsqueeze(1), values).squeeze(1) #(b x 1 x src_len) · (b x src_len x 2h) = (b x 1 x 2h)
        #Todo:
        #   1. output c_i: input representation
        #   2. ouput attention coefficients
        U_t = torch.cat((query, c_t), 1)    
        V_t = self.combined_output_projection(U_t)
        out = torch.tanh(V_t)
        return out

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, src_len: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.
        mask[i,j] = 1 on padding, batch i, symbol j
        """
        #Do once per batch, outside the decoding loop
        self.mask = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float, device = self.device)
        for i, len in enumerate(src_len): self.mask[i, len:] = 1

class myAttention_ver_1(nn.Module):
    def __init__(self, attType, enc_hidden_size, dec_hidden_size, device):
        super().__init__()
        #v_t = W_u [a_t; h_t^dec]
        self.mask = None
        self.device = device
        if attType == 'dot':                self.attFn = self.attDot
        elif attType == 'mult':   
            self.attFn = self.attMultiplicative
            self.attMat = nn.Linear(enc_hidden_size, dec_hidden_size,False, device)
        elif attType == 'add':         
            self.attFn = self.attAdditive
            self.attMat = nn.Linear(enc_hidden_size, dec_hidden_size,True, device)
        else: raise Exception(f"attType {attType} is not recognized must be 'dot', 'mult' or 'add'")
    def forward(self, 
                    values: TensorType['batch_size', 'src_len', 'enc_hidden_size'], 
                    keys:   TensorType['batch_size', 'src_len', 'dec_hidden_size'], 
                    query:  TensorType['batch_size', 'dec_hidden_size']
                    ) ->    Tuple[Tuple, torch.Tensor, torch.Tensor]: #recibe generadores
        """ Compute one forward step of the LSTM decoder, attention computation."""
        # alpha_i,t = softmax( k*q )
        # c_t = alpha_i,t v_i
        # U_t = [query; a_t]
        # out = tanh(W U_t)

        #Similarity function k*q
        e_t = self.attFn(keys,query)

        # Set e_t to -inf where enc_masks has 1 (zeros pads)
        if self.mask is not None: e_t.data.masked_fill_(self.mask.bool(), float('-inf'))

        alpha_t = F.softmax(e_t, dim = 1)
        c_t = torch.bmm(alpha_t.unsqueeze(1), values).squeeze(1) #(b x 1 x src_len) · (b x src_len x 2h) = (b x 1 x 2h)
        #Todo:
        #   1. output c_i: input representation
        #   2. output attention coefficients
        return c_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, src_len: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.
        mask[i,j] = 1 on padding, batch i, symbol j
        """
        #Do once per batch, outside the decoding loop
        self.mask = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float, device = self.device)
        for i, len in enumerate(src_len): self.mask[i, len:] = 1
    def attDot(self, keys, query):
        return torch.bmm(keys, query.unsqueeze(-1)).squeeze(2)  #(b x src_len x h) · (b x h x 1) = (b x src_len x 1)
    def attMultiplicative(self, keys:   TensorType['batch_size', 'key_len','hidden_size'], 
                                query:  TensorType['batch_size', 'hidden_size']):
        query = self.attMat(query)
        return torch.bmm(keys, query.unsqueeze(-1)).squeeze(2)  #(b x src_len x h) · (b x h x 1) = (b x src_len x 1)

    def attAdditive(self, keys, query):
        # query = self.attMat(query)
        # return torch.bmm(keys, query.unsqueeze(-1)).squeeze(2)  #(b x src_len x h) · (b x h x 1) = (b x src_len x 1)
        # torch.tanh()
        pass