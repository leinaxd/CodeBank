from typing import Union
import pandas as pd

class sentenceSplit:
    """
    Splits a string into n-words delimited by space

    <n_words>: number of words concatenated
    <overlap>: Porcentage of overlapped words
    """
    def __init__(self, n_words:int, overlap:float=0, power=True):
        assert 0 < n_words, f"n_words must be positive"
        assert 0<=overlap and overlap<1, f"overlap must be a number between [0 and 1)"
        self.delimiter  = ' '
        self.n_words = n_words
        self.offset = int(overlap*n_words)
        # self.min_len = self.n_words*0.5 #at least do half window
        self.min_len = self.n_words*overlap #at least do half window
        self.power = power
    def splitString(self, txt:str)-> list:
        txt = txt.split(self.delimiter)
        result = []
        s=0
        if len(txt) <= self.min_len: result.append(' '.join(txt)) #for texts smaller than the window
        while s < len(txt)-self.min_len:
            e = s + self.n_words
            result.append(' '.join(txt[s:e]))
            s = e-self.offset
        return result

    def __call__(self, data:Union[pd.Series, str]) -> Union[pd.DataFrame, list]:
        if self.power==False: return data
        if isinstance(data, str):         return self.splitString(data)
        elif isinstance(data, pd.Series): return data.apply(self.splitString)
        else:                             raise NotImplementedError





if __name__ == '__main__':
    test = 5

    
    if test == 1:
        print(f"test {test} manual overlapping")
        data = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        groups = 10
        overlap = 0.75
        offset = round(overlap*groups)
        print(f"groups: {groups}\noverlap: {overlap}, offset = {offset}\ndata:\n{data}")
        s = 0
        while s < len(data):
            e = s+groups
            print(data[s:e])
            s = e-offset

    if test == 2:
        print(f"test {test}: overlapping test")
        txt = 'Estaba caminando por el parque hasta que me encontré un cachorrito, era tan tierno que me lo llevé a mi casa. He notado que este cachorrito tenía dueño y era justamente de aquella persona que andaba persiguiendo aquel día.'
        for o in [0,0.4,0.99,1]:
            splitter = sentenceSplit(10,o)
            print(f"{'='*50}\noverlapping: {o}\n{splitter(txt)}\n{'='*50}\n")

    if test == 3:
        print(f"test {test}: data frame test")
        data = pd.DataFrame({'lower':['a b c d e f g h i j','aa bb cc dd ee ff gg hh ii jj'], 
                             'upper':['A B C D E F G H I J','AA BB CC DD EE FF GG HH II JJ']})
        splitter = sentenceSplit(4,0.5)
        out = splitter(data['upper'])
        data['upper4']=out
        print(data.T)


    if test == 4:
        print(f"test {test}: last split is useless. check the last item isn't already cover by the last but one item\n")
        txt = 'Estaba caminando por el parque hasta que me encontré un cachorrito, era tan tierno que me lo llevé a mi casa. He notado que este cachorrito tenía dueño y era justamente de aquella persona que andaba persiguiendo aquel día.'
        for o in [0, 0.5, 0.9]:
            print(f"overlapping of: {o}")
            splitter = sentenceSplit(4,o)
            out = splitter(txt)
            print(out,'\n')

    if test == 5:
        print(f"test {test}: fewer strings than splits\n{'='*50}")
        txt = 'A B C D E F G H I J'
        n_words = 20
        overlap = 0.5
        splitter = sentenceSplit(n_words,overlap)
        out = splitter(txt)
        print(f"Params:\n\tn_words:{n_words}\n\toverlap:{overlap}")
        print(f"INPUT:\n\t{txt}\n\tlen={len(txt.split(' '))}")
        print(f"OUTPUT:\n\t{out}")
        print(f"{'='*50}\nEXPECTED:\n\t{txt}")




