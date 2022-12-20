import pandas as pd
from typing import Union, List

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import logging

import nlpaug.augmenter.word as naw

# import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
# import nlpaug.augmenter.sentence as nas
# import nlpaug.flow as nafc
# from nlpaug.util import Action
import nltk
import torch

from CodeBank.DatasetFramework.DataAugmentation.transformations import txtRandomSampling

class txtSynonyms:
    """
    <model_path>: wordnet / huggingFace model
    sources:    
        - https://towardsdatascience.com/powerful-text-augmentation-using-nlpaug-5851099b4e97
        - https://github.com/makcedward/nlpaug
        - https://github.com/makcedward/nlpaug/blob/master/example/textual_language_augmenter.ipynb
    """

    def __init__(self, model_path='wordnet', prob=0.3, customModel=False):
        nltk.download('omw')

        if model_path == 'wordnet':
            self.aug = naw.SynonymAug(aug_src='wordnet', lang='spa',aug_p=prob)
        else:    
            self.aug = naw.ContextualWordEmbsAug(model_path=model_path, aug_p=prob)
            # model_path='bert-base-multilingual-uncased'
            # model_path='dccuchile/bert-base-spanish-wwm-uncased' #BETO
        if customModel:
            doSoftmax=False
            self.doSoftmax=doSoftmax
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            self.count  = 0
            self.max_length=512
            logging.set_verbosity_error() #Ignore unused weights warning. (this model is for finetunning)
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)
            logging.set_verbosity_warning()
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.sampling = txtRandomSampling(self.transformation['synonyms'], maskToken=self.tokenizer.mask_token)        
    # def __init__(self, transformations:dict, HuggingFaceLM:str, maskToken:str):
    #     self.sampling = txtRandomSampling(transformations['synonyms'], maskToken)
    #     self.modelLM = HuggingFaceLM

    def __call__(self, txt:Union[str,List[str]]): 
        return self.aug.augment(txt)


    def doSynonyms(self, txt:str):
        """Custom implementation"""
        if self.verbose: 
            self.count+=1
            print(f"{self.count}",end='')
        input_txt = self.sampling(txt)
        input = self.tokenizer(input_txt, return_tensors = 'pt', truncation=True,max_length=self.max_length)
        input = {k:v.to(self.device) for k, v in input.items()}

        input_ids = input['input_ids'][0] #ignore batching dim
        masked_ix, = torch.where(input_ids == self.tokenizer.mask_token_id)

        self.model.eval()
        predictions = self.model(**input).logits

        for ix in masked_ix.tolist():
            # predicted_token = self.selection(predictions[0,ix])
            mask_prob = predictions[0,ix]
            if self.doSoftmax: mask_prob = torch.nn.functional.softmax(mask_prob, dim=-1)
            predicted_token = torch.multinomial(mask_prob, 1)
            input_ids[ix] = predicted_token

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1:-1])
        txt = self.tokenizer.convert_tokens_to_string(tokens)
        return txt

if __name__ == '__main__':
    test = 1
    if test == 1:
        data = pd.DataFrame({'src':
            ['Un rápido zorro marrón salta sobre un perro perezoso', 
             'Estaba perdido, caminé en dirección opuesta al sol y me encontré en una ciudad fantasma espeluznante']})
        aug = txtSynonyms()
        print( data['src'] )
        print( aug(data['src'].tolist()) )