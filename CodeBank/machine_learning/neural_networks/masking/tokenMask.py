import torch, torch.nn as nn
from torchtyping import TensorType

class tokenMask(nn.Module):
    def __init__(self, token):
        super().__init__()
        self.token = token
    def forward(self, src: TensorType['batch_size', 'seq_len']
                    )  ->  TensorType['batch_size', 'seq_len']:
        mask = (src != self.token)
        return mask


if __name__=='__main__':
    maxLen, dataset_size, seqLen, pad = 10, 2, 4, 0
    src = torch.randint(1, maxLen, size=(dataset_size, seqLen), requires_grad=False)
    src[:, 0] = 1
    src[:, -1] = pad
    mask = tokenMask(0)
    print(f'src| {src.shape}\n{src}')
    print(mask(src))