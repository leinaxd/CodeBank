from typing import Callable
import pandas as pd

class dataEstimator:
    """
    Applies an estimator to a pandas Series
    """

    def __init__(self, estimator:Callable[...,dict]):
        self.estimator = estimator
    def __call__(self, data:pd.Series):
        result = {}
        for sample in data:
            out = self.estimator(sample)
            for k,v in out.items():
                if k not in result: result[k] = []
                result[k].append(v)
        return pd.DataFrame(result)



if __name__ == '__main__':
    test = 1

    data = pd.DataFrame({
        'a':[1,2,3,4,5,6,7,8],
        'b':['uno '*5,'dos '*5,'tres '*5,'cuatro '*5,'cinco '*5,'seis '*5,'siete '*5,'ocho '*5],
        })

    if test == 1:
        print(f"test {test}: dataframe")
        def count(txt): 
            return {'char':txt[:3],'len':len(txt)}
        estimator = dataEstimator(count)
        out = estimator(data['b'])
        print(out)
        data = pd.concat((data,out),1)
        print(data)

    # if test == 2:
    #     print(f"test {test} estimating the mean")
    #     def estimator(txt):
    #         return txt