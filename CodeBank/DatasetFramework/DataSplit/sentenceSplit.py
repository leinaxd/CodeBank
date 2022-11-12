from typing import Union
import pandas as pd

class sentenceSplit:
    """
    Splits a string into n-words delimited by space

    <n_words>: number of words concatenated
    <overlap>: Porcentage of overlapped words
    """
    def __init__(self, n_words:int, overlap:float=0):
        assert 0<=overlap and overlap<=1, f"overlap must be a number between 0 and 1"
        self.delimiter  = ' '
        self.n_words = n_words
        self.overlap = overlap

    def splitString(self, txt:str)-> list:
        txt = txt.split(self.delimiter)
        result = []
        numIxs = int(len(txt) // (self.n_words*self.overlap))
        for i in range(numIxs):
            s = int(i*self.n_words*self.overlap)
            e = s + self.n_words
            result.append(' '.join(txt[s:e]))
        return result

    def __call__(self, data:Union[pd.DataFrame, str]) -> Union[pd.DataFrame, list]:
        if isinstance(data, str): return self.splitString(data)
        else:
            raise NotImplementedError





if __name__ == '__main__':
    test = 1
    if test == 1:
        print(f"test {test}: split a text into a list of words")
        txt = 'Estaba caminando por el parque hasta que me encontré un cachorrito, era tan tierno que me lo llevé a mi casa. He notado que este cachorrito tenía dueño y era justamente de aquella persona que andaba persiguiendo aquel día.'
        splitter = sentenceSplit(10,0.4)
        print(splitter(txt))
    if test == 2:
        print(f"test {test}: split a text into a list of strings")
        data = pd.DataFrame('asd')