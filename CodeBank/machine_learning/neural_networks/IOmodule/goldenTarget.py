
import torch
from torchAPI import torchModule
from ..masking import padSeq
from ..lossCriterion import loglikelihood

class goldenTarget(torchModule):
    """
    TASK
        apply the <model>,
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
        self.endToken = endToken
        self.device = torch.device('cpu')
        self.padBatch = padSeq(padToken)
        self.lossCriterion = loglikelihood()
        self.padToken = padToken

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
        tgt, tgt_len = self.padBatch(tgt)  #a = batch x answer_len

        prediction, pred_len = self.forward(seq, seqLen, tgt, tgt_len)
        
        loglikelihood = self.lossCriterion(prediction, tgt)

        # print('train:', pred_len, ans_len)
        # reg = torch.abs(pred_len-ans_len.to(self.device)).to(self.device)
        # reg2 = torch.square(pred_len-ans_len).to(self.device)

        #Zero out padding probabilities (they're outside the generated sequence)
        tgt_masks = (tgt != self.padToken).float()
        loglikelihood = loglikelihood*tgt_masks
        # loglikelihood = 100*loglikelihood/(pred_len[:,None].expand(loglikelihood.shape))
        
        #log probability of generating true target words
        return -loglikelihood.sum(dim=1) #loss = -logP(w1...wn)
        # return -loglikelihood.sum(dim=1)+reg #loss = -logP(w1...wn)
        # return -loglikelihood.sum(dim=1)+reg2 #loss = -logP(w1...wn)

    def forward(self, seq, seq_len, tgt, tgt_len): raise NotImplementedError

