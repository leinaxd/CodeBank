"""
TODO:
    loss criterion in different module
    - KLdivergence
    - loglikelihood
    Input:
        tgt_len
        prob_smoothing
    Output:
        loss
"""
from typing import List, Tuple
from torchtyping import TensorType
import torch, torch.nn as nn
from CodeBank.NeuralNetworks.masking import padSeq, padMask
from CodeBank.NeuralNetworks.masking import labelSmoothing

from CodeBank.NeuralNetworks.encoders import transformerEncoder
from .categoryDecoder import categoryDecoder

from CodeBank.NeuralNetworks.regularizers.pretraining import maskPretraining

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from CodeBank.NeuralNetworks.outputLayer import logSoftmax

class TransformerModel(nn.Module):
    """
    pPreMask: [0.8]
        probability of masking a token in the pretrain stage
    """
    special_tokens = ['<s>','<pad>','</s>','<ukn>','<mask>']
    @classmethod
    def make_vocab(self, path:List[str], vocab_size=30_000, min_frequency=2, show_progress=True, save=False):
        """
        path: [output dir, file1, file2 ...]
        """
        
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(path[1:], vocab_size, min_frequency, show_progress, cls.special_tokens)
        if save: tokenizer.save_model(path[0], '')
        return tokenizer
    
    def __init__(self,  dirPath:    str,
                        embed_size: int, ff_size:    int,
                        nLayers:    int, nHeads:     int, 
                        pDropout: float
                        ):
        super().__init__()

        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:    self.tokenizer = ByteLevelBPETokenizer(dirPath+'-vocab.json', dirPath+'-merges.txt')
        except: raise FileNotFoundError(f"tokenizer vocab not found, please call {__class__.__name__}.make_vocab([corpus1, corpus2 ...]) to build one")

        padToken = self.tokenizer.token_to_id('<pad>')
        uknToken = self.tokenizer.token_to_id('<ukn>')
        sosToken = self.tokenizer.token_to_id('<s>')
        eosToken = self.tokenizer.token_to_id('</s>')
        self.tokenizer._tokenizer.post_processor = BertProcessing(
                                                            ('<s>', sosToken),
                                                            ('</s>', eosToken),
                                                            )



        vocab_size = self.tokenizer.get_vocab_size()
        self.padSeq  = padSeq(padToken)
        self.padMask = padMask()
        
        self.encoder = transformerEncoder(vocab_size, embed_size, ff_size, nLayers, nHeads, pDropout)
        # self.decoder = categoryDecoder(output_size, embed_size, ff_size, nLayers, nHeads, pDropout)
        self.pretrainedDecoder = logSoftmax(embed_size, vocab_size)

        # self.lossMask = labelSmoothing(confidence) #confidence=0.9)
        self.lossCriterion = nn.KLDivLoss(reduction='none')

        # self.pretrain = False
        # self.maskPretraining = maskPretraining(pPreMask,self.preField,uknToken)

    def forward(self, src:list) -> TensorType['batch_size', 'output_size']:
        src, src_len = self.padSeq(src)
        srcRepr      = self.encoder(src, src_len)

        # if self.pretrain: prediction = self.pretrainDecoder(srcRepr)
        # else:             prediction = self.decoder(srcRepr, src_len)

        # return prediction

    def predict(self, batch    #pd.DataFrame
                    ) -> list: 
        src        = batch[self.srcField].to_list()
        prediction = self.forward(src)
        prediction = torch.argmax(prediction, -1)
        prediction = prediction.tolist()
        return prediction

    def update(self, batch):
        raise NotImplementedError
        loss = self.loss(batch)
        return loss
    
    def loss(self, batch:Tuple[List[str], List[str]]):
        src = [self.tokenizer.encode(src).ids for src in batch[0]]
        tgt = [self.tokenizer.encode(tgt).ids for tgt in batch[1]]
        # tgt = self.tokenizer(tgt)
        # print('src: ',src['ids'])
        print([len(s) for s in src])
        print([len(s) for s in tgt])
        # for s in src['ids']:

        # raise NotImplementedError
        loss = self.model(src)
        return loss
    
    def loss(self, batch   #pd.DataFrame   #Dict[str, pd.DataFrame]
                    ) ->    TensorType['batch_size']:
        if self.pretrain: return self.pretrainedLoss(batch)
        else:             return self.decoderLoss(batch)

    def decoderLoss(self, batch   #pd.DataFrame
                    ) ->    TensorType['batch_size']:
        src        = batch[self.srcField].to_list()
        prediction = self.forward(src)
        tgt          = batch[self.tgtField].to_list()
        tgt_smoothed = self.lossMask(prediction, tgt)

        loss = self.lossCriterion(prediction, tgt_smoothed)
        loss = torch.sum(loss, -1) #sum over vocab
        # print(loss.shape)
        return loss

    def pretrainedLoss(self, 
                       batch #pd.DataFrame
                    ) ->    TensorType['batch_size']:
        # a = batch[self.preField]
        # a.to_list()
        src, tgt   = self.maskPretraining(batch)
        src        = src[self.preField].to_list()
        prediction = self.forward(src)   #batch x seq_len x output_size

        tgt        = tgt[self.preField].to_list()
        tgt, tgt_len = self.padSeq(tgt)

        tgt_smoothed = self.lossMask(prediction, tgt)

        loss = self.lossCriterion(prediction, tgt_smoothed)

        tgtMask      = self.padMask(tgt_len).unsqueeze(-1)
        loss.masked_fill_(~tgtMask, 0) #ignore padding positions
        loss = torch.sum(loss, -1) #sum over vocab
        loss = torch.sum(loss, -1)/tgt_len #sum over seq

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