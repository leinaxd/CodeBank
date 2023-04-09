import torch
from ..torchModule import torchModule
from typing import Tuple
from torchtyping import TensorType

class seq2seq(torchModule):
    """
    TASK
        Given an input sequence of variable length i apply the <model>,
        process the 

    Ex.
        class myModel(seq2seq):
            def forward(self, seq, seq_len):
                return seq, seq_len
        m = myModel()
        m.predict(batch)
        m.loss(batch)
    Specification
        Model: torchModule
        Input: Sequence of Variable Length
        Output: Sequence of Variable Length
    """
    def __init__(self, padToken:int, endToken:int):
        super().__init__()
        self.padToken = padToken
        self.endToken = endToken
        self.device = torch.device('cpu')

    def padBatch(self, batchedSequence) -> Tuple[TensorType['batch_size','seq_len',int], TensorType['batch_size',1,int]]:
        """pads a variable length sequence into a fixed tensor"""
        seq_len = [len(sequence) for sequence in batchedSequence] 
        max_length = max(seq_len)
        seq = [sequence + [self.padToken]*(max_length-len(sequence)) for sequence in batchedSequence]
        seq = torch.tensor(seq, dtype=torch.long, device=self.device)# Tensor: (b, seq_len)
        seq_len = torch.tensor(seq_len) 
        return seq, seq_len #seq.to('cuda'), seq_len.to('cpu')

    def unpadBatch(self, batchedSequence, batchLen):
        return 3

    def predict(self, batch):
        seq = batch['x']
        seq, seq_len = self.padBatch(seq)

        prediction, pred_len = self.forward(seq, seq_len)
        #Greedy strategy: (best of each word)
        prediction = torch.argmax(prediction, dim=-1)

        prediction = [p[: pred_len[i]-1].tolist() for i, p in enumerate(prediction)] #no padToken, no endToken
        return prediction

    def loss(self, batch):
        seq, tgt = batch['x'], batch['y']
        # c, q, a      = trainBatch
        # seq = [c[i]+q[i] for i in range(len(c))]
        seq, seqLen = self.padBatch(seq)
        for i in tgt: i.append(self.endToken) 
        ans, ans_len = self.padBatch(tgt)  #a = batch x answer_len

        prediction, pred_len = self.forward(seq, seqLen)
        # prediction,pred_len = self.forward(seq, (ans, ans_len))

        if prediction.shape[1] < ans.shape[1]:
            zeros = torch.zeros([prediction.shape[0], ans.shape[1]-prediction.shape[1], prediction.shape[2]], device=self.device)
            prediction = torch.cat((prediction, zeros),dim=1)
        loglikelihood  = torch.gather(prediction, index=ans.unsqueeze(-1), dim=-1).squeeze(-1)

        # print('train:', pred_len, ans_len)
        # reg = torch.abs(pred_len-ans_len.to(self.device)).to(self.device)
        # reg2 = torch.square(pred_len-ans_len).to(self.device)

        #Zero out padding probabilities (they're outside the generated sequence)
        tgt_masks = (ans != self.padToken).float()
        loglikelihood = loglikelihood*tgt_masks
        # loglikelihood = 100*loglikelihood/(pred_len[:,None].expand(loglikelihood.shape))
        
        #log probability of generating true target words
        return -loglikelihood.sum(dim=1) #loss = -logP(w1...wn)
        # return -loglikelihood.sum(dim=1)+reg #loss = -logP(w1...wn)
        # return -loglikelihood.sum(dim=1)+reg2 #loss = -logP(w1...wn)

    def forward(self, seq, seq_len): raise NotImplementedError



#TODO:
#   Las secuencias se combinan desde el dataset
