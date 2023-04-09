
import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Tuple

class resRNN(nn.Module):
    """
    El objetivo es que se vea como una sola unidad LSTM
    es como una pared entre depth y stacks

    input:          input[t],  hidden[t]
    output:         hidden[t+1]

    hidden_depth:   number of layers
    residual:       gradient explosion solution out = f(input) + input
    output_size:    rnn output state if required
    nStacks:        num of stacked resRNN where in[k+1] = hidden[k] (Alias nLayers deprecated)
    depth:          num stacked ResRNN running in parallel
    """
    #TODO: bidirectional, multidirectional?
    def __init__(self, input_size, hidden_size, nDepth=1, nStacks = 1, residual =False, residualStack=False, device = None):
        #Obs. h=output, c=hidden
        super().__init__()
        assert hidden_size%nDepth*nStacks==0, 'total_hidden_size must be a multiple of depth and nStacks'
        self.nStacks = nStacks
        self.nDepth = nDepth
        self.hidden_size = hidden_size
        self.unit_hidden_size = int(hidden_size/(nStacks*nDepth))
        self.device=device
        self.residual = residual #through time
        self.residualStack = residualStack #through layers
        #Creates a wall (depth x stack) of recurrents units
        self.units=[]
        # self.residualStack=False
        for i in range(nDepth):
            self.units.append([])
            for j in range(nStacks):
                if self.residualStack: x = input_size
                else: 
                    x = input_size if j == 0 else self.unit_hidden_size
                self.units[i].append( nn.LSTMCell(x, self.unit_hidden_size,device=self.device) )
        if self.residualStack: self.stacks_proj = nn.Linear(self.unit_hidden_size, input_size)
        self.nMem=2

    def initialize(self, dec_state:  TensorType['nMem','nStacks','batch_size','hidden_size',float]):
        batch_size = dec_state[0].shape[1]
        nMem = dec_state.shape[0]
        self.dec_state: TensorType['nMem','nDepth','nStacks','batch_size','unit_hidden_size'] \
                    = torch.empty([nMem,self.nDepth,self.nStacks,batch_size, self.unit_hidden_size])
        # dec_state = torch.stack(dec_state,dim=0)
        for i in range(self.nDepth):
            hx = dec_state[0,:,:,i*self.unit_hidden_size:(i+1)*self.unit_hidden_size]
            cx = dec_state[1,:,:,i*self.unit_hidden_size:(i+1)*self.unit_hidden_size]
            self.dec_state[0,i,:,:,:] = hx
            self.dec_state[1,i,:,:,:] = cx

    def get_state(self):
        raise NotImplementedError
        hx = self._squeezeDepth(0)
        cx = self._squeezeDepth(1)
        return (hx,cx)

    def forward(self,   input:      TensorType['batch_size', 'input_size'], 
                        state:      TensorType['nMem','nStacks','batch_size','hidden_size']
                        ) ->        TensorType['batch_size', 'hidden_size']:
        # batch_size = state.shape[2]
        newState = torch.zeros(state.shape, device=self.device) #OJO CON INICIALIZAR EN EMPTY!
        for i in range(self.nDepth):
            s = i*self.unit_hidden_size
            e = (i+1)*self.unit_hidden_size
            for j in range(self.nStacks):
                if j == 0: dec_input = input 
                else: dec_input = dec_input+self.stacks_proj(h_out) if self.residualStack else h_out
                # else: dec_input = h_out
                hx = state[0,j,:,s:e]
                cx = state[1,j,:,s:e]
                h_out,c_out = self.units[i][j](dec_input, (hx, cx))
                newState[0,j,:,s:e] = h_out+hx if self.residual else h_out
                # newState[1,j,:,s:e] = c_out
                newState[1,j,:,s:e] = c_out #It goes to +inf if you c_out+cx a memory
                # print(torch.max(newState[0,j,:,s:e]))

                # assert not torch.isinf(state).any().item(), 'state = NAN'
                # assert not torch.isinf(dec_input).any().item(), 'dec_input = NAN'
                # assert not torch.isinf(hx).any().item(), 'hx = NAN'
                # assert not torch.isinf(h_out).any().item(), 'h_out = NAN'
                # assert not torch.isinf(cx).any().item(), 'cx = inf'
                # assert not torch.isinf(c_out).any().item(), 'c_out = inf'
                # assert not torch.isinf(h_out+hx).any().item(), 'hx+cout = NAN'
                # print(torch.max(hx), torch.max(cx))
                # assert not torch.isinf(c_out+cx).any().item(), f'cx+cout = inf, {torch.max(c_out)}, {torch.max(cx)}'
                # assert not torch.isinf(newState).any().item(), 'newState = NAN'
        ho   = newState[0,-1,:,:]
        return ho, newState
        
    def _squeezeDepth(self, nMem):
        """Returns current last state. 
            0:hidden, 
            1:cell"""
        #Concatenate depth vector through hidden dimension
        ho = torch.empty((self.dec_state.shape[3],self.hidden_size))
        for i in range(self.nDepth):
            s = i*self.unit_hidden_size
            e = (i+1)*self.unit_hidden_size
            ho[:,s:e] = self.dec_state[nMem,i,-1,:,:]
        return ho
    

class unitRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super().__init__()
        self.nMem=1
        self.linear1=nn.Linear(input_size+hidden_size, hidden_size, device=device)
        self.linear2=nn.Linear(hidden_size, hidden_size, device=device)
        self.linear3=nn.Linear(hidden_size, hidden_size, device=device)
        self.sigma=torch.tanh
    def forward(self, x, h):
        out=torch.cat((x,h),1)
        out=self.linear1(out)
        out=self.sigma(out)
        out=self.linear2(out)
        out=self.sigma(out)
        out=self.linear3(out)
        return out
        
class resRNN_v2(nn.Module):
    """
    El objetivo es que se vea como una sola unidad LSTM
    es como una pared entre depth y stacks

    input:          input[t],  hidden[t]
    output:         hidden[t+1]

    hidden_depth:   number of layers
    residual:       gradient explosion solution out = f(input) + input
    output_size:    rnn output state if required
    nStacks:        num of stacked resRNN where in[k+1] = hidden[k] (Alias nLayers deprecated)
    depth:          num stacked ResRNN running in parallel
    """
    #TODO: bidirectional, multidirectional?
    def __init__(self, input_size, hidden_size, nDepth=1, nStacks = 1, residual =False, residualStack=False, device = None):
        #Obs. h=output, c=hidden
        super().__init__()
        assert hidden_size%nDepth*nStacks==0, 'total_hidden_size must be a multiple of depth and nStacks'
        self.nStacks = nStacks
        self.nDepth = nDepth
        self.hidden_size = hidden_size
        self.unit_hidden_size = int(hidden_size/(nStacks*nDepth))
        self.device=device
        self.residual = residual #through time
        self.residualStack = residualStack #through layers
        #Creates a wall (depth x stack) of recurrents units
        self.units=[]
        for i in range(nDepth):
            self.units.append([])
            for j in range(nStacks):
                if self.residualStack: x = input_size
                else: 
                    x = input_size if j == 0 else self.unit_hidden_size
                self.units[i].append( unitRNN(x, self.unit_hidden_size,device=self.device) )
        if self.residualStack: self.stacks_proj = nn.Linear(self.unit_hidden_size, input_size)
        self.nMem=1

    def forward(self,   input:      TensorType['batch_size', 'input_size'], 
                        state:      TensorType['nMem','nStacks','batch_size','hidden_size']
                        ) ->        TensorType['batch_size', 'hidden_size']:
        # batch_size = state.shape[2]
        newState = torch.zeros(state.shape, device=self.device) #OJO CON INICIALIZAR EN EMPTY!
        for i in range(self.nDepth):
            s = i*self.unit_hidden_size
            e = (i+1)*self.unit_hidden_size
            for j in range(self.nStacks):
                if j == 0: dec_input = input 
                else: dec_input = dec_input+self.stacks_proj(h_out) if self.residualStack else h_out
                hx = state[0,j,:,s:e]
                h_out = self.units[i][j](dec_input, hx)
                newState[0,j,:,s:e] = h_out+hx if self.residual else h_out
        ho   = newState[0,-1,:,:]
        return ho, newState
        
    def _squeezeDepth(self, nMem):
        """Returns current last state. 
            0:hidden, 
            1:cell"""
        #Concatenate depth vector through hidden dimension
        ho = torch.empty((self.dec_state.shape[3],self.hidden_size))
        for i in range(self.nDepth):
            s = i*self.unit_hidden_size
            e = (i+1)*self.unit_hidden_size
            ho[:,s:e] = self.dec_state[nMem,i,-1,:,:]
        return ho
    