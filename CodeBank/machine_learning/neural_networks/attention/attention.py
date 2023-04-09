
import math
import torch, torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType



class attScores(nn.Module):
    """
    Computes the 'Scaled Dot Product Attention'
    WARNING:
        if a Mask left uncover an entire row, 
        then a nan value will be computed,
        this is because of softmax([0,0,0]) = [nan, nan, nan]

        In order to avoid backward nan, the first mask col will always be True
    """
    def __init__(self, pDropout:int):
        super().__init__()
        self.dropout = nn.Dropout(p=pDropout)
        self.sigma = nn.Softmax(dim=-1)
        # self.sigma = nn.ReLU()
        # torch.autograd.set_detect_anomaly(True)
        # self.verbose = False
    def forward(self,   query:  TensorType['batch_size', 'nHeads', 'q_len', 'embed_size/nHeads'], 
                        key:    TensorType['batch_size', 'nHeads', 'k_len', 'embed_size/nHeads'], 
                        mask:   TensorType['batch_size', 'seq_len','seq_len']=None
                        )   ->  TensorType['batch_size', 'nHeads', 'q_len', 'k_len']: 
        dim_k = query.size(-1)
        #score = (b x h x q_len x emb) x (b x h x emb x k_len) = (b x h x q_len x k_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)

        # Same mask applied to all h heads.
        if mask is not None: 
            mask[:,:,0]=True #Force at least one True value (The first)
            mask = mask.unsqueeze(1) #nHeads
            scores.masked_fill_(mask.logical_not(), float('-inf'))
        alpha_t = self.sigma(scores)
        # alpha_t = torch.nan_to_num(alpha_t, nan=0) #nan = 0
        alpha_t = self.dropout(alpha_t)
        # if self.verbose: print(alpha_t)
        return alpha_t


class attention(nn.Module):
    """
    Input: 
        Query: Sequence of embeddings
        Key:   Sequence of embeddings
        Value: Sequence of embeddings
    Output:
        Convex combination of Values
        out_t = sum_i alpha_i,t v_i

    Matricial form:
                   ---- t ---->
        ↓ [o1]   [a11, a12, .. a1t] [v1]
        ↓ [o2]   [a21, a22, .. a2t] [v2]
        i [o3] = [a31, a32, .. a3t] [v3]
        ↓ [..]   [       ..       ] [..]
        ↓ [ot]   [ai1, ai2, .. ait] [vt]

    Articles:
        https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a#:~:text=In%20layman's%20terms%2C%20the%20self,these%20interactions%20and%20attention%20scores.
    """
    def __init__(self, pDropout:int):
        super().__init__()
        self.scores = attScores(pDropout)
    def forward(self, query:TensorType['batch_size', 'nHeads', 'seq_len', 'embed_size/nHeads'], 
                        key:TensorType['batch_size', 'nHeads', 'seq_len', 'embed_size/nHeads'], 
                      value:TensorType['batch_size', 'nHeads', 'seq_len', 'embed_size/nHeads'], 
                       mask:TensorType['batch_size', 'seq_len']=None
                        )-> TensorType['batch_size', 'nHeads', 'seq_len', 'embed_size/nHeads']: 
        alpha_t=self.scores(query, key, mask)
        attention = torch.matmul(alpha_t, value)
        # print(alpha_t)
        # print(value)
        return attention



if __name__=='__main__':
    test = 1
    if test == 1:
        print('='*50)
        print(f"test {test}: matmul with nan")
        print('='*50)
        a = torch.tensor([[1],[10],[100]])
        b = torch.tensor([[0,1,2]])
        print(f"a x b:\n{a}\n\tx\n{b}\n----------------\n{torch.matmul(a,b)}")
        print('='*50)
        a = torch.tensor([[1.],[10],[100]])
        b = torch.tensor([[0,1,torch.nan]])
        print(f"a x b:\n{a}\n\tx\n{b}\n----------------\n{torch.matmul(a,b)}")
        print('='*50)
        a = torch.tensor([[1.],[10],[torch.nan]])
        b = torch.tensor([[0,1,2.]])
        print(f"a x b:\n{a}\n\tx\n{b}\n----------------\n{torch.matmul(a,b)}")      
        print('='*50)
        a = torch.tensor([[0.,1],[2,3]])
        b = torch.tensor([[1, torch.nan],[2,3]])
        print(f"\n{a}\n\tx\n{b}\n----------------\n")
        print(f"a x b:\n {torch.matmul(a,b)}")
        print(f"b x a:\n {torch.matmul(b,a)}")
        print('='*50)
        a = torch.tensor([[1.],[10],[torch.nan]])
        print(a.shape)
        l = nn.Linear(1,3)
        print(f"a:\n{a}\nlinear(a):\n{l(a)}")

    if test == 2:
        mask = torch.tensor([[True, False, False],[True, True, False], [True, True, True]])
        data = torch.tensor([[1,2,3],[4,5,6],[7,8,9.]])
        score = torch.masked_fill(data, ~mask, float('-inf'))
        print(score)
    if test == 3:
        data = torch.FloatTensor([ [[ 1, 2, 3, 4],
                                    [-1,-2,-3,-4]],
                                   [[ 5, 6, 7, 0],
                                    [-5,-6,-7, 0]],
                                   [[-8, 0, 0, 0],
                                    [ 8, 0, 0, 0]]])
        data.transpose_(1,2)
        mask = torch.tensor([[True, True, True, True],
                             [True, True, True, False],
                             [True, False, False, False]])
        mask.unsqueeze_(-1)
        print(data.shape)
        print(mask.shape)
        score = torch.masked_fill(data, ~mask, float('-inf'))
        print(score)
        print(mask)
    if test == 4:
        data = torch.FloatTensor([[[ 10, 20, 30, 40],
                                   [  1,  2,  3,  4],
                                   [ -1, -2, -3, -4],
                                   [-10,-20,-30,-40]],
                                  [[ 50, 60, 70,  0],
                                   [  5,  6,  7,  0],
                                   [ -5, -6, -7,  0],
                                   [-50,-60,-70,  0]],
                                  [[ 80,  0,  0,  0],
                                   [  8,  0,  0,  0],
                                   [ -8,  0,  0,  0],
                                   [-80,  0,  0,  0]]])
        # data.transpose_(1,2)
        data_len = [4,3,1]
        mask = torch.zeros_like(data, dtype=torch.bool)
        print(data.shape)
        print(mask.shape)
        for i, len in enumerate(data_len): 
            mask[i, len:, :]=True
            mask[i, :, len:]=True
        print(mask)
        # mask = torch.ones([len(data), seq_len, seq_len]) #batch_size, seq_len x sec_len

        score = torch.masked_fill(data, mask, float('-inf'))
        print(score)
        # print(mask)
        