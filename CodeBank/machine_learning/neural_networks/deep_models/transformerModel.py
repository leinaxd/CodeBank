
from typing import List
from torchtyping import TensorType
import torch, torch.nn as nn

from CodeBank.machine_learning.neural_networks.masking import padSeq, padMask
# from CodeBank.machine_learning.neural_networks.lossCriterion import KLDsmoothing
from CodeBank.machine_learning.neural_networks.masking import padSeq, padMask

from CodeBank.machine_learning.neural_networks.encoders import transformerEncoder
from CodeBank.machine_learning.neural_networks.decoders import transformerDecoder
# from CodeBank.machine_learning.neural_networks.decoders import goldenDecoder
from CodeBank.machine_learning.neural_networks.decoders import heuristicDecoder
from CodeBank.machine_learning.neural_networks.heuristicSearch import greedyStrategy

from CodeBank.machine_learning.neural_networks.tokenization import placeToken
from CodeBank.machine_learning.neural_networks.masking import labelSmoothing
from CodeBank.DataFramework.DataVisualization import ContinuousPlot
from torch.optim import Adam

class TransformerModel(nn.Module):
    """
    Transformer Model 
    Paper:
        - AttentionIsAllYouNeed
            https://arxiv.org/abs/1706.03762
    <input_size> size of the input vocab
    <output_size> size of the output vocab
    pPreMask: [0.8]
        probability of masking a token in the pretrain stage
    """
    def __init__(self,  input_size: int, output_size:int, 
                        embed_size: int, ff_size:    int,
                        nLayers:    int, nHeads:     int, 
                        padToken:   int, startToken: int, endToken: int, 
                        pDropout: float, confidence: float
                        ):
        super().__init__()
        self.EOS = placeToken(endToken, -1)
        self.BOS = placeToken(startToken, 0)
        self.padSeq  = padSeq(padToken)
        self.padMask = padMask()
        
        self.encoder = transformerEncoder(input_size, embed_size, ff_size, nLayers, nHeads, pDropout)
        self.decoder = transformerDecoder(output_size, embed_size, output_size, ff_size, nLayers, nHeads, pDropout)
        # self.trainDecoder   = goldenDecoder(decoder)

        heuristic = greedyStrategy()
        self.predictDecoder = heuristicDecoder(self.decoder, heuristic, startToken, endToken)

        self.lossMask       = labelSmoothing(confidence)
        self.lossCriterion  = nn.KLDivLoss(reduction='none')

        self.optimizer = Adam(self.parameters(), lr=1E-3)
        # optimizer = Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9)
        # model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, optimizer)

    def update(self, src:List[int], tgt:List[int]):
        loss = self.loss(src, tgt)
        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
      
    def predict(self, src:list) -> list: 
        """Prediction for generative task"""
        src, src_len = self.padSeq(src)
        srcRepr      = self.encoder(src, src_len)

        prediction, pred_len = self.predictDecoder(srcRepr, src_len)
        prediction = prediction.tolist()
        # prediction = [prediction[i][:pred_len[i]-1] for i in range(len(prediction))] #No EOS
        # prediction = [prediction[i][1:pred_len[i]-1] for i in range(len(prediction))] #No BOS and EOS
        #POSIBLE BUG
        #   Si el sistema genera un startToken, el sistema podría no decodificarlo
        #   -> Solucion... Tokens no reconocidos, sale como <ukn>
        return prediction
    

    
    def loss(self, src:List[int], tgt:List[int]) -> TensorType['batch_size']:
        """loss for generative task"""
        src, src_len = self.padSeq(src)
        srcRepr      = self.encoder(src, src_len)
        
        dec_input          = tgt
        dec_input          = self.BOS(dec_input)   #input = tgt shifted SOS
        dec_input, tgt_len = self.padSeq(dec_input)

        prediction         = self.decoder(dec_input, tgt_len, srcRepr, src_len)
        
        tgt          = self.EOS(tgt)   #tgt with EOS
        tgt, tgt_len = self.padSeq(tgt)
        tgt_smoothed = self.lossMask(prediction, tgt)
        tgtMask      = self.padMask(tgt_len).unsqueeze(-1) #mask all vocab corresponding to pad

        loss = self.lossCriterion(prediction, tgt_smoothed)
        loss.masked_fill_(~tgtMask, 0) #ignore padding positions
        
        loss = torch.sum(loss, -1) #sum over vocab
        loss = torch.sum(loss, -1)/tgt_len #sum over seq
        return loss
    
    #[CLASSIFICATION TASK]    
    def predict_classification(self, src:list) -> list: 
        """Prediction for classifation task"""
        prediction = self.forward(src)
        prediction = torch.argmax(prediction, -1)
        prediction = prediction.tolist()
        return prediction
    def decoderLoss(self, src, tgt) ->    TensorType['batch_size']:
        """loss for 1st vector classification task"""
        prediction = self.forward(src)
        loss = self.lossCriterion(prediction, tgt)
        loss = torch.sum(loss, -1) #sum over vocab
        return loss
    def pretrainedLoss(self, src, tgt) -> TensorType['batch_size']:
        """loss for attentive-classification task"""
        prediction = self.forward(src)   #batch x seq_len x output_size
        tgt, tgt_len = self.padSeq(tgt)
        loss = self.lossCriterion(prediction, tgt)
        tgtMask      = self.padMask(tgt_len).unsqueeze(-1)
        loss.masked_fill_(~tgtMask, 0) #ignore padding positions
        loss = torch.sum(loss, -1) #sum over vocab
        loss = torch.sum(loss, -1)/tgt_len #sum over seq
        return loss


if __name__ == '__main__':
    test = 2
    if test == 1:
        print(f"Test {test}: Generative loss\n")
        src = [[1,2,3,4],[5,6,7],[1,2]]
        tgt = [[4,2,1],[2,3,1],[5,5]] 
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = TransformerModel(
            input_size=8, output_size=6,
            embed_size=128, ff_size=128,
            nLayers=2, nHeads=2, 
            padToken=0, startToken=0, endToken=0, 
            pDropout=0, confidence=1
            )
        
        loss = model.loss(src,tgt)
        print(loss)
    if test == 2:
        print(f"Test {test}: Generative train\n")
        vocab_size = 11
        model = TransformerModel(
            input_size=vocab_size, output_size=vocab_size,
            embed_size=128, ff_size=128,
            nLayers=2, nHeads=2, 
            padToken=0, startToken=0, endToken=0, 
            pDropout=0, confidence=1
            )

        def data_gen(vocab_size=11, seq_len=10, batch_size=4, nbatches=10):
            "Generate random data for a src-tgt copy task."
            for _ in range(nbatches):
                src = torch.randint(1, vocab_size, size=(batch_size, seq_len))
                src[:, 0] = 1 #BOS
                tgt = src.clone()
                yield src.tolist(), tgt.tolist()

        plot = ContinuousPlot()
        loss_hist = []
        max_epochs = 50
        x = list(range(max_epochs))
        for epoch in x:
            model.train()
            total_loss = 0
            nSamples = 0
            for src, tgt in data_gen():
                nSamples += 1
                loss = model.update(src, tgt)
                total_loss += loss 
            loss_hist.append(total_loss/nSamples)
            plot(x[:epoch+1],loss_hist)