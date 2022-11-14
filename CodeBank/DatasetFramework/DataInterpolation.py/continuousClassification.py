import pandas as pd
from CodeBank.DatasetFramework.DataSplit import sentenceSplit

class continuousClassification:
    """ 
    1. Splits the data into n-subsequences
    2. Apply the estimator in each step
    3. return a sequence of appling the estimator to each subsequence
    """

    def __init__(self, n_words, overlap, estimatorFunc:callable, srcField:str):
        self.splitter = sentenceSplit(n_words, overlap)
        self.estimator = estimatorFunc
        self.srcField = srcField
    def __call__(self, data:pd.DataFrame):
        data[self.srcField]
        out = self.splitter(data)

        out = self.estimator(out)

        if isinstance(out, dict):   #añadir columnas
            pass
        data = data
        return data


if __name__ == '__main__':
    test = 1
    data = pd.DataFrame({
        'a':[1,2,3,4,5,6,7,8],
        'b':['uno','dos','tres','cuatro','cinco','seis','siete','ocho'],
        })

    if test == 1:
        print(f"test {test}: dataframe")
        evol = continuousClassification(3,0,None,'b')
        data = {}