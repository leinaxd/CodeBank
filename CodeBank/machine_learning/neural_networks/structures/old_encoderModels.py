#Author: Eichenbaum, Daniel
#Encoders Libraries
#   Includes:
#       LSTM Encoder  
#   Features:
#       Variable length input sequence
from typing import List, Tuple
from torch.functional import Tensor
from torchtyping import TensorType
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchAPI.networkFramework.seq2seq.recurrentUnits import resRNN, resRNN_v2

class Encoder_ver_standford(nn.Module):
    """LSTM Encoder with padded sequence"""
    def __init__(self, input_size, embed_size, hidden_size, padToken=None, bidirectional=False,):
        super().__init__()
        self.srcEmb = nn.Embedding(input_size, embed_size, padToken)
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=bidirectional)
    def forward(self, src: torch.Tensor, src_len:List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
        orignal src (implicit): variable length sequence
        src:                    Padded tensor                       #(batch, max_seq_length)
        src_len:                original length of each sequence    #(batch, 1)
        """
        src = self.srcEmb(src) #Input
        src = pack_padded_sequence(src, src_len, enforce_sorted=True) #Empaqueto para ahorrar computo
        enc_hiddens, (last_hidden, last_cell) = self.encoder(src)   #Aplico el encoder Bi-LSTM
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, True)   #Lo desempaqueto en una matriz para utilizar
        last_hidden = torch.cat(tuple(last_hidden), 1) #Tuple, corta por la ultima dimension
        last_cell   = torch.cat(tuple(  last_cell), 1)
        return enc_hiddens, (last_hidden, last_cell)

class myEncoder_ver_1(nn.Module):
    """LSTM Encoder with padded sequence"""
    def __init__(self, embed_size, hidden_size, bidirectional=False, nLayers=1, device = None):
        super().__init__()
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers = nLayers, batch_first=True, bidirectional=bidirectional, device=device)

    def forward(self,   seq:              TensorType['batch_size', 'seq_len', 'embed_size',int], 
                        seq_len:          TensorType['batch_size', 1, int],
                        init_state: Tuple[TensorType['num_layers','batch_size','hidden_size'],
                                          TensorType['num_layers','batch_size','hidden_size']]=None
                        ) ->        Tuple[          TensorType['batch_size','seq_len','hidden_size'], 
                                            Tuple[  TensorType['num_layers','batch_size','hidden_size'],
                                                    TensorType['num_layers','batch_size','hidden_size']]]:
        seq = pack_padded_sequence(seq, seq_len, batch_first=True, enforce_sorted=False) #Empaqueto para ahorrar computo
        seq_hiddens, last_state = self.encoder(seq, init_state) #Aplico el encoder Bi-LSTM
        seq_hiddens, _ = pad_packed_sequence(seq_hiddens, True)
        return seq_hiddens, last_state

class myEncoder_ver_2(nn.Module):
    """LSTM Encoder with multihead
        -padded sequence
        -same hidden_size but different depth (multiple RNN units concatenated as a single hidden layer)
    Nota:
        -Poner el hidden_size en paralelo hace independiente la evolución de los hidden_states
        Es decir,
            -1 bloque hidden_size al realimentarse se interconectan todos los hx con todos los hx
            -2 bloques hidden_size/2 al realimentarse consigue evolucionar 2 hx de forma independiente entre sí
        El objetivo es que la red aprenda diferentes representaciones, es el backpropagation quien especializa luego
    """
    def __init__(self, embed_size, total_hidden_size, bidirectional=False, nLayers=1, depth=1, device=None):
        super().__init__()
        assert total_hidden_size%depth==0, 'total_hidden_size must be a multiple of depth'
        self.device = device
        self.depth = depth
        self.total_hidden_size = total_hidden_size
        self.hidden_size = int(total_hidden_size/depth)
        self.numLayers = nLayers
        self.encoders = {i:nn.LSTM(embed_size, self.hidden_size, num_layers = nLayers, batch_first=True, bidirectional=bidirectional, device=device) for i in range(depth)}
    def forward(self,   seq:              TensorType['batch_size', 'seq_len', 'embed_size',int], 
                        seq_len:          TensorType['batch_size', 1, int],
                        init_state: Tuple[TensorType['num_layers','batch_size','hidden_size'],
                                          TensorType['num_layers','batch_size','hidden_size']]=None
                        ) ->        Tuple[          TensorType['batch_size','seq_len','hidden_size'], 
                                            Tuple[  TensorType['num_layers','batch_size','hidden_size'],
                                                    TensorType['num_layers','batch_size','hidden_size']]]:
        batch_size = seq.shape[0]
        total_seq_hiddens = torch.empty([batch_size, seq.shape[1], self.total_hidden_size],device=self.device)
        total_last_hidden = torch.empty([self.numLayers, batch_size, self.total_hidden_size],device=self.device)
        total_last_cell   = torch.empty([self.numLayers, batch_size, self.total_hidden_size],device=self.device)

        seq = pack_padded_sequence(seq, seq_len, batch_first=True, enforce_sorted=False) #Empaqueto para ahorrar computo
        for i in range(self.depth):
            seq_hiddens, (last_hidden, last_cell) = self.encoders[i](seq, init_state) #Aplico el encoder Bi-LSTM
            seq_hiddens, _ = pad_packed_sequence(seq_hiddens, True)
            total_seq_hiddens[:,:,i*self.hidden_size:(i+1)*self.hidden_size] = seq_hiddens
            total_last_hidden[:,:,i*self.hidden_size:(i+1)*self.hidden_size] = last_hidden
            total_last_cell[:,:,i*self.hidden_size:(i+1)*self.hidden_size]   = last_cell
        return total_seq_hiddens, (total_last_hidden, total_last_cell)

class myEncoder_ver_3(nn.Module):
    """LSTM Encoder with multihead
        -padded sequence
        -same hidden_size but different depth (multiple RNN units concatenated as a single hidden layer)
    Nota:
        -Poner el hidden_size en paralelo hace independiente la evolución de los hidden_states
        Es decir,
            -1 bloque hidden_size al realimentarse se interconectan todos los hx con todos los hx
            -2 bloques hidden_size/2 al realimentarse consigue evolucionar 2 hx de forma independiente entre sí
        El objetivo es que la red aprenda diferentes representaciones, es el backpropagation quien especializa luego
    """
    def __init__(self, embed_size, hidden_size, nStack=1, nDepth=1, recurrentUnit='LSTM', residual=True,device=None):
        super().__init__()
        self.device = device
        self.nStack = nStack
        self.nDepth = nDepth
        self.hidden_size = hidden_size
        if recurrentUnit == 'LSTM':
            self.encoder = resRNN(embed_size, hidden_size,nDepth,nStack,residual,residual,device)
        elif recurrentUnit == 'resRNN':
            self.encoder = resRNN_v2(embed_size, hidden_size,nDepth,nStack,residual,residual,device)
        else: raise Exception(f"recurrentUnit {recurrentUnit} unknow, try 'LSTM' or 'resRNN'")
        self.nMem = self.encoder.nMem
    def forward(self,   seq:              TensorType['batch_size', 'seq_len', 'embed_size',int], 
                        seq_len:          TensorType['batch_size', 1, int],
                        init_state: TensorType['nMem','nStacks','batch_size','hidden_size']=None,
                        ) ->        Tuple[          TensorType['batch_size','seq_len','hidden_size'], 
                                            Tuple[  TensorType['num_layers','batch_size','hidden_size'],
                                                    TensorType['num_layers','batch_size','hidden_size']]]:
        batch_size = seq.shape[0]
        if init_state==None: init_state = torch.zeros([self.nMem,self.nStack, batch_size,self.hidden_size], device=self.device)
        # self.encoder.initialize(init_state)
        enc_state = init_state

        total_hiddens = torch.zeros([batch_size, max(seq_len),self.hidden_size], device = self.device)
        #enforce sorted:
        _,indices = torch.sort(seq_len)
        e_len = 0
        for i, ix in enumerate(indices):
            s_len = e_len
            e_len = seq_len[ix]
            for l in range(s_len, e_len): #apply len times
                longest_batchs_ix = indices[i:]
                
                #choose not-ended batches
                input = seq[longest_batchs_ix,l,:] #(batch x embedding)
                state = enc_state[:,:,longest_batchs_ix,:] #a fraction of the state

                hx, state = self.encoder(input, state)

                #reasemble
                enc_state[:,:,longest_batchs_ix,:] = state #update only a fraction of the state
                total_hiddens[longest_batchs_ix, l, :] = hx

        # last_state = self.encoder.get_state()
        final_state=[]
        for i in range(self.nMem):
            final_state.append(enc_state[i,:,:,:])
        return total_hiddens, final_state
