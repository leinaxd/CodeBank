
import torch, torch.nn as nn
from torchtyping import TensorType


class PureCorrelation(nn.Module):
    """
    Testing the attention mechansim without Q, K, V matrices

    It does just a simple dot product correlation
    """
    def __init__(self, embed_size, pDropout):
        super().__init__()
        self.linear = nn.Linear(embed_size, embed_size)
        self.linear_2 = nn.Linear(embed_size, embed_size)
        # self.sigma = nn.Softmax(dim=-1)
        self.sigma = nn.ReLU()
        self.dropout = nn.Dropout(pDropout)
    def forward_2(self, query:TensorType['batch_size', 'seq_len', 'embed_size'], 
                      key:TensorType['batch_size', 'seq_len', 'embed_size'], 
                      value:TensorType['batch_size', 'seq_len', 'embed_size'], 
                      mask:TensorType['batch_size', 'seq_len']=None
                      )-> TensorType['batch_size','seq_len','embed_size']: 
        key = self.linear(key)
        value = self.linear_2(value)
        out = self.sigma( key + value )
        print(key.shape)
        print(value.shape)
        print(out.shape)
        out = self.dropout(out)
        return out
                
    def forward(self, query:TensorType['batch_size', 'seq_len', 'embed_size'], 
                        key:TensorType['batch_size', 'seq_len', 'embed_size'], 
                      value:TensorType['batch_size', 'seq_len', 'embed_size'], 
                       mask:TensorType['batch_size', 'seq_len']=None
                        )-> TensorType['batch_size','seq_len','embed_size']: 
        
        #M = (batch_size x seq_len x embed_size) · (embed_size x embed_size)· (batch_size x embed_size x seq_len)
        #Q = (batch_size x seq_len x embed_size) · (batch_size x embed_size x seq_len) = (batch_size x seq_len x seq_len)
        key = self.linear(key) #Mahalanovich distance
        out = torch.matmul(query, key.transpose(-2, -1))
        # alpha_t = self.sigma(out)
        if mask is not None: 
            mask[:,:,0]=True #Force at least one True value (The first)
            out.masked_fill_(mask.logical_not(), float('-inf'))

        alpha_t = self.sigma(out)
        alpha_t = self.dropout(alpha_t)

        value = self.linear_2(value)
        out = torch.matmul(alpha_t, value)# sigma(X*X) · X

        return out
