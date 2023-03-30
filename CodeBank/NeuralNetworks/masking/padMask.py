import torch, torch.nn as nn
from torchtyping import TensorType
from typing import Tuple
import torch.nn.functional as F

class padMask(nn.Module):
    """
    A squared mask usefull for attention
    <dim>: output shape / seq_len
    <data_len>: The sizes of every sample (filled with pad)
    """
    def forward(self, *data_len: Tuple[TensorType['batch_size']]
                         )  ->  TensorType['batch_size','dim']:
        device = data_len[0].device
        batch_size = data_len[0].size(0)

        maxSizes = [torch.max(data) for data in data_len]
        mask = torch.ones([batch_size]+maxSizes, dtype=torch.bool, device=device)
        for i in range(batch_size):
            sample = mask[i]
            for j, len in enumerate(data_len):
                l = torch.arange(len[i], maxSizes[j], device=device)
                sample.index_fill_( j, l,False)
            # if dim==1:
                # mask[i, len:] = False
            # elif dim==2:
                # mask[i, len:, :]=False
                # mask[i, :, len:]=False
            # else: raise NotImplementedError(f'For dim={self.dim}')
        return mask


if __name__=='__main__':
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
    data_len = torch.tensor([4,3,1])
    # data.transpose_(1,2)

    test = 3
    print('Usualy attention requires defines a matrix. Whose padded tokens goes to -inf')
    print('test 1) You can define an 1D mask and later ignore padded output\n\n')
    print('test 2) You can try an usual 2D mask with NaN after softmax at pads positions')
    print('test 3) 2D mask but diferents shapes (for encoder-decoder alignament)')
    print(f'Test: {test}')
    if test == 1:
        mask = padMask()
        mask = mask(data_len) #1D mask
        mask.unsqueeze_(1)
        print('data.shape',data.shape)
        print('mask.shape',mask.shape)
        print(mask)
        score = torch.masked_fill(data, ~mask, float('-inf'))
        print('score\n',score)
        print('softmax\n',F.softmax(score, -1))
    if test == 2:
        mask = padMask() 
        mask = mask(data_len, data_len) #2D mask
        print('data.shape',data.shape)
        print('mask.shape',mask.shape)
        print(mask)
        score = torch.masked_fill(data, ~mask, float('-inf'))
        print('score\n',score)
        print('softmax\n',F.softmax(score, -1))
    if test == 3:
        mask = padMask() 
        ydata_len = data_len
        xdata_len = torch.tensor([2,1,1])
        data = data[:,:2]
        print(f"xdata_len: {xdata_len}, ydata_len: {ydata_len}")
        mask = mask(xdata_len, ydata_len) #2D mask diferent sizes
        print(mask)
        print('data.shape',data.shape)
        print('mask.shape',mask.shape)
        score = torch.masked_fill(data, ~mask, float('-inf'))
        print('score\n',score)
        print('softmax\n',F.softmax(score, -1))
    if test == 4:
        maxLen, dataset_size, seqLen, pad = 10, 2, 4, 0
        src = torch.randint(1, maxLen, size=(dataset_size, seqLen), requires_grad=False)
        src[:, 0] = 1
        src[:, -1] = pad
        mask = padMask()
        print(f'src| {src.shape}\n{src}')
        print(mask(src))
        