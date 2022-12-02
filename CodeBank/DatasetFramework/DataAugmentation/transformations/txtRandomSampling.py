from typing import Union
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

class txtRandomSampling:
    """
    Samples the corpus and place the "[MASK]" token with probability p

    <prob> probability of mask a word
    """
    nltk.download('punkt')
    def __init__(self, prob:float, maskToken:str):
        self.prob = prob
        self.maskToken = maskToken
    def doTxt(self, corpus:str):
        words = word_tokenize(corpus)
        ix = np.random.binomial(1,self.prob,len(words))
        words = [self.maskToken if i else w for w,i in zip(words,ix)]
        return ' '.join(words)

    def doDataFrame(self, data:pd.Series):
        return data.apply(self.doTxt)
        # return data

    def __call__(self, txt:Union[str,pd.Series]):
        # print(' '.join(words))
        # ix = list(range(len(words)))
        # print(ix)
        # print(random.choices(words))
        if isinstance(txt, str): return self.doTxt(txt)
        if isinstance(txt, pd.Series): return self.doDataFrame(txt)





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
        transformation = txtRandomSampling(0.1,'[MASK]')
        out = transformation(corpus)
        print(out)

    if test == 2:
        print(f"test {test}: masking a dataframe")
        data = pd.DataFrame({'corpus':[corpus, 'This is a second corpus'], 'type':['history','other']})
        transformation = txtRandomSampling(0.5,'[MASK]')
        out = transformation(data['corpus'])
        print(out)