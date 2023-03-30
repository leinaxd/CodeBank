from typing import Tuple, Sequence
from torchtyping import TensorType
import torch, torch.nn as nn

class padSeq(nn.Module):
    """
    padSeq:
        Given an input batch of sequences of variable length.
        Returns a fixed length tensor, and the lengths associatied with every sample

        input:  
            batch = List[List[int]]
            batch = pandas.Series[List[int]]
        output: 
            seq     = Tensor['batch_size', 'seq_len']
            seq_len = Tensor['batch_size']
    """
    def __init__(self, padToken):
        super().__init__()
        self.padToken = padToken
        self.register_buffer('device', torch.empty([0])) # a trick for getting current device
    def forward(self, batch: Sequence[Sequence[int]]
                )  -> Tuple[ TensorType['batch_size','seq_len',int], 
                             TensorType['batch_size',1,int]]:
        """pads a variable length sequence into a fixed tensor"""
        # device = self.device.device
        device = self.device.device

        seq_len = [len(sequence) for sequence in batch] 
        max_length = max(seq_len)
        seq = [sequence + [self.padToken]*(max_length-len(sequence)) for sequence in batch]
        seq = torch.tensor(seq, dtype=torch.long, device=device)# Tensor: (b, seq_len)
        seq_len = torch.tensor(seq_len, device=device) 
        return seq, seq_len #seq.to('cuda'), seq_len.to('cpu')

    def unpadBatch(self, batchedSequence, batchLen):
        return 3