import pandas as pd


from typing import Dict
from torchtyping import TensorType
import torch, torch.nn as nn


from ...masking import padSeq, padMask


from ...masking import labelSmoothing

from ...encoders import transformerEncoder

# from ...decoders import goldenDecoder
from ...decoders import heuristicDecoder
from ...decoders import transformerDecoder
from ...heuristicSearch import greedyStrategy

from ...tokenization import placeToken

# from torchAPI import torchModule

class AttentionIsAllYouNeed(nn.Module):
    def __init__(self,  input_size: int, output_size:int, 
                        embed_size: int, ff_size:    int,
                        nLayers:    int, nHeads:     int, 
                        padToken:   int, startToken: int, endToken: int, 
                        pDropout: float, confidence: float,
                        srcField = 'src', tgtField = 'tgt',
                        ):
        super().__init__()
        self.srcField = srcField
        self.tgtField = tgtField
        self.EOS = placeToken(endToken, -1)
        self.SOS = placeToken(startToken, 0)
        self.padSeq  = padSeq(padToken)
        self.padMask = padMask()
        
        self.encoder = transformerEncoder(input_size, embed_size, ff_size, nLayers, nHeads, pDropout)
        self.decoder = transformerDecoder(output_size, embed_size, output_size, ff_size, nLayers, nHeads, pDropout)

        # self.trainDecoder   = goldenDecoder(decoder)
        heuristic = greedyStrategy()
        self.predictDecoder = heuristicDecoder(self.decoder, heuristic, startToken, endToken)

        self.lossMask = labelSmoothing(confidence) #confidence=0.9)
        self.lossCriterion = nn.KLDivLoss(reduction='none')

    def predict(self, batch:    pd.DataFrame
                    ) -> list: 
        src          = batch[self.srcField].to_list()
        src, src_len = self.padSeq(src)
        srcRepr      = self.encoder(src, src_len)

        prediction, pred_len = self.predictDecoder(srcRepr, src_len)
        prediction = prediction.tolist()
        # prediction = [prediction[i][:pred_len[i]-1] for i in range(len(prediction))] #No EOS
        # prediction = [prediction[i][1:pred_len[i]-1] for i in range(len(prediction))] #No SOS and EOS
        #POSIBLE BUG
        #   Si el sistema genera un startToken, el sistema podría no decodificarlo
        #   -> Solucion... Tokens no reconocidos, sale como <ukn>
        return prediction

    def loss(self, batch:   pd.DataFrame   #Dict[str, pd.DataFrame]
                    ) ->    TensorType['batch_size']:
        src          = batch[self.srcField].to_list()
        src, src_len = self.padSeq(src)
        srcRepr      = self.encoder(src, src_len)
        
        dec_input          = batch[self.tgtField].to_list()
        dec_input          = self.SOS(dec_input)   #input = tgt shifted SOS
        dec_input, tgt_len = self.padSeq(dec_input)

        prediction         = self.decoder(dec_input, tgt_len, srcRepr, src_len)
        
        tgt          = batch[self.tgtField].to_list()
        tgt          = self.EOS(tgt)   #tgt with EOS
        tgt, tgt_len = self.padSeq(tgt)
        tgt_smoothed = self.lossMask(prediction, tgt)
        tgtMask      = self.padMask(tgt_len).unsqueeze(-1)

        loss = self.lossCriterion(prediction, tgt_smoothed)
        loss.masked_fill_(~tgtMask, 0) #ignore padding positions
        
        loss = torch.sum(loss, -1) #sum over vocab
        loss = torch.sum(loss, -1)/tgt_len #sum over seq
        # print(loss.shape)
        return loss
        

 
if __name__ == '__main__':
    import numpy as np
    test = 1
    if test == 1:
        src = [[1,2,3,4],[5,6,7],[1,2]]
        tgt = [[4,2,1],[2,3,1],[5,5]] 
        print('forward')
    elif test == 2:
        # def data_gen(V, batch, nbatches):
        #     "Generate random data for a src-tgt copy task."
        #     for i in range(nbatches):
        #         src = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        #         src[:, 0] = 1
        #         tgt = src.clone()
        #         yield Batch(src, tgt, 0)
        # V = 11
        # criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        # model = make_model(V, V, N=2)
        # model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        # for epoch in range(10):
        #     model.train()
        #     run_epoch(data_gen(V, 30, 20), model, 
        #             SimpleLossCompute(model.generator, criterion, model_opt))
        #     model.eval()
        #     print(run_epoch(data_gen(V, 30, 5), model, 
        #                     SimpleLossCompute(model.generator, criterion, None)))
        pass



#TODO
#   -> IOmodule
#       padBatch devuelve tipo de dato tensor=(seq, seq_len)
#       luego operaciones en el tensor, son operaciones en seq limitados por seq_len
#   -> Crear carpeta
#       IOmodule vs LOSS MODELS
#       - ECM
#       - loglikelihood
#       - cross entropy
#   -> Aproximar softmax
#   -> Unificar seq y seq_len
#   -> Diferentes callbacks para "eval" y "train", además forward si aplica una sola funcion

#TODO:
# refactorizar en transformerEncoder


