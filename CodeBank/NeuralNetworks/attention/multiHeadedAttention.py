import torch.nn as nn
from torchtyping import TensorType

try:
    from .attention import attention
except:
    from attention import attention

class multiHeadedAttention(nn.Module):
    """
    <key>:   The information selector vector
    <value>: The information source vector
    <query>: The information query vector
    linear mapping:
        q = Q·query
        k = K·key
        v = V·value
    embed_size: 
        size of the linear mapping
        Q, K, V
    """
    def __init__(self, nHeads, embed_size, pDropout):
        super().__init__()
        assert embed_size % nHeads == 0
        # We assume d_v always equals d_k
        self.dim_k = embed_size // nHeads
        self.h = nHeads
        self.linear_1 = nn.Linear(embed_size, embed_size)
        self.linear_2 = nn.Linear(embed_size, embed_size)
        self.linear_3 = nn.Linear(embed_size, embed_size)
        self.linear_4 = nn.Linear(embed_size, embed_size)
        self.attention = attention(pDropout)

        
    def forward(self, query:TensorType['batch_size', 'seq_len', 'embed_size'], 
                        key:TensorType['batch_size', 'seq_len', 'embed_size'], 
                      value:TensorType['batch_size', 'seq_len', 'embed_size'], 
                       mask:TensorType['batch_size', 'seq_len']=None
                        )-> TensorType['batch_size','seq_len','embed_size']: 
        
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.linear_1(query).view(batch_size, -1, self.h, self.dim_k).transpose(1,2)
        key   = self.linear_2(key  ).view(batch_size, -1, self.h, self.dim_k).transpose(1,2)
        value = self.linear_3(value).view(batch_size, -1, self.h, self.dim_k).transpose(1,2)
        
        # 2) Apply attention on all the projected vectors in batch. 
        out = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear. 
        #out:        (batch x nHeads x seq_len x embed_size/nHeads)
        #transpose:  (batch x seq_len x nHeads x embed_size/nHeads)
        #contiguous: (batch x seq_len x nHeads x embed_size/nHeads)
        #view:       (batch x seq_len x embed_size)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.dim_k)       
        out = self.linear_4(out)
        # import torch
        # torch.nan_to_num_(out, nan=0)
        return out

if __name__ == '__main__':
    from torchAPI.networkFramework.masking import padMask, padSeq
    import torch, torch.nn as nn
    embed_size = 4
    data = [[1,2,3],[4,5]] #batch = 2, seq_len = (3,2)

    print(f"data:\n{data}\nembed_size:{embed_size}\n{'='*50}")

    test = 1
    if test == 1:
        print(f"test {test}: nan in attention\n{'='*50}")
        #define
        mask = padMask()
        pad  = padSeq(0)
        emb = nn.Embedding(6,embed_size)
        att = multiHeadedAttention(2,embed_size,0)

        #apply
        src, src_len = pad(data)
        print(f"padded src:\n{src}\nsrc_len:\n{src_len}\n{'-'*50}")
        src  = emb(src)
        print(f"embedded src:\n{src}\nsrc.shape:\n{src.shape}\n{'='*50}")
        mask = mask(src_len, src_len)
        print(f"mask:\n{mask}\n{'-'*50}")
        out = att(src,src,src,mask)
        print(f"out:\n{out}\n{'='*50}")
        out = att(out,out,out,mask)
        print(f"twice out:\n{out}\n{'-'*50}")
        
    if test == 2:
        a = torch.tensor([[1.,2,3],[1,2,torch.nan]], requires_grad=True)
        out_1 = a+1
        out_2 = torch.sum(out_1)
        out_2.backward()
        print(a)
        print(out_1)
        print(out_2)
        print(a.grad)
        