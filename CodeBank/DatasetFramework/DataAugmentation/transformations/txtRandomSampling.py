from typing import Union
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

nltk.download('punkt')

class txtRandomSampling:
    """
    Samples the corpus and place the "[MASK]" token with probability p

    <prob> probability of mask a word
    """
    def __init__(self, prob:float, maskToken:str,return_ix=False):
        self.prob = prob
        self.maskToken = maskToken
        self.return_ix = True
    def doTxt(self, corpus:str):
        words = word_tokenize(corpus)
        ix = np.random.binomial(1,self.prob,len(words))
        ix = ix.nonzero()[0]
        # words = [self.maskToken if i else w for w,i in zip(words,ix)]
        for i in ix: words[i] = self.maskToken #update with maskToken
        
        txt = ' '.join(words)
        return ix, txt

    def doDataFrame(self, data:pd.Series):
        ix=[]
        txt = []
        for sample in data:
            _ix, _txt = self.doTxt(sample)
            ix.append(_ix)
            txt.append(_txt)
        return pd.Series(ix), pd.Series(txt)

    def __call__(self, txt:Union[str,pd.Series]) -> (pd.Series, pd.Series):
        # print(' '.join(words))
        # ix = list(range(len(words)))
        # print(ix)
        # print(random.choices(words))
        if isinstance(txt, str): out = self.doTxt(txt)
        if isinstance(txt, pd.Series): out = self.doDataFrame(txt)
        
        if self.return_ix: return out
        else: return out[1] #return txt only






if __name__ == '__main__':
    test = 2
    corpus = """
    El Mesozoico, era mesozoica o era secundaria, 
    conocido zoológicamente como la era de los dinosaurios o 
    botánicamente como la era de las cícadas, 
    es una división de la escala temporal geológica 
    que pertenece al eón Fanerozoico; 
    dentro de este, el Mesozoico sigue al Paleozoico y precede al Cenozoico, 
    de ahí su nombre, que procede del griego μεσο que significa "entre", 
    y ζώον, que significa "de los animales" que significa "vida intermedia". 
    Se inició hace 251 millones de años y finalizó hace 66 millones de años"""
    if test == 1:
        print(f"test {test}: masking a text")
        transformation = txtRandomSampling(0.1,'[MASK]',return_ix=True)
        ix, out = transformation(corpus)
        print(ix)
        print(out)

    if test == 2:
        print(f"test {test}: masking a dataframe")
        data = pd.DataFrame({'corpus':[corpus, 'This is a second corpus'], 'type':['history','other']})
        transformation = txtRandomSampling(0.3,'[MASK]',return_ix=True)
        ix, out = transformation(data['corpus'])
        print(ix)
        print(out)