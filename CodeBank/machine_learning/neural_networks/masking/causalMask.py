from torchtyping import TensorType
import torch, torch.nn as nn

class causalMask(nn.Module):
    def __init__(self, diagonal=0):
        super().__init__()
        self.diagonal = diagonal
    def forward(self, tgt:TensorType['batch_size', 'seq_len']):
        # mask = (tgt != self.tokenPad).unsqueeze(-2)
        device = tgt.device
        size = tgt.size(-1)
        attn_shape = (tgt.size(0), size, size)
        causalMask = torch.ones(attn_shape, dtype=torch.bool, device=device)
        causalMask = torch.tril(causalMask,self.diagonal) #triangular lower
        # ntokens = (tgt_y != self.tokenPad).data.sum()
        return causalMask

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test = 1
    if test == 1:
        maxLen = 10
        dataset_size = 2
        seqLen = 4
        tgt = torch.randint(1, maxLen, size=(dataset_size, seqLen), requires_grad=False)
        pad = 0
        tgt[:, 0] = 1
        tgt[:, -1] = pad
        print(f'tgt| {tgt.shape}\n{tgt}')
        tgt = tgt[:, :-1] #Quito el ultimo simbolo
        tgt_y = tgt[:, 1:] #Quito el primer simbolo
        mask = causalMask(0)
        tgt_mask = mask(tgt)
        data = torch.ones([2,3])
        fig = plt.figure(figsize=(5,5))
        ax1=plt.subplot(1,2,1)
        ax2=plt.subplot(1,2,2)
       
        b = mask(data)
        print(b)
        ax1.imshow(b[0])
        ax2.imshow(b[1])
        plt.show()