from typing import Callable
import pandas as pd
from CodeBank.DatasetFramework.DataSplit import sentenceSplit
from CodeBank.DatasetFramework.DataEstimation import dataEstimator

class segmentatedClassification:
    """ 
    Recipe
    1. Splits the data into n-subsequences
    2. Apply the estimator in each step
    3. return a sequence of appling the estimator to each subsequence
    """
    # raise NotImplementedError('lo divido en sentenceSplitter y data estimation')
    def __init__(self, n_words, overlap, srcField:str):
        self.splitter = sentenceSplit(n_words, overlap)
        self.estimator = dataEstimator()
        self.srcField = srcField

    def __call__(self, data:pd.DataFrame, estimatorFunc:Callable[..., dict]):
        out = self.splitter(data[self.srcField])
        newData = self.estimator(out, estimatorFunc)
        return pd.concat((data,newData),1)



if __name__ == '__main__':
    test = 1
    data = pd.DataFrame({
        'a':[1,2,3,4,5,6,7,8],
        'b':['uno '*5,'dos '*5,'tres '*5,'cuatro '*5,'cinco '*5,'seis '*5,'siete '*5,'ocho '*5],
        })

    if test == 1:
        print(f"test {test}: dataframe")
        def estimator(txt): 
            return {'parte_1':txt[0],'parte_2':txt[1]}
        hist = segmentatedClassification(3,0,estimator,'b')
        out = hist(data)
        print(out)
        