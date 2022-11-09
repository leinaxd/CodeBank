from typing import Tuple
import pandas as pd
try:
    from datasetSplit import datasetSplit
except:
    from . import datasetSplit


class bootStrap:
    """
    Performs the bootstrap evaluation loop

    parameters:
        <categoryField> the dataset column for sample in a balanced manner
        <nTrain>/<nTest> proportion of the splitted dataset
    1. split the dataset into train-val in a balanced fashion
    2. returns tuples of train-val
    3. Repeat <sample_size> times


    __call__(dataset:pd.DataFrame):
        yield train_dataset, test_dataset

    Paper:
        "Bootstrap methods for standard errors, confidence intervals and other methods of statistical accuracy"
        https://projecteuclid.org/journals/statistical-science/volume-1/issue-1/Bootstrap-Methods-for-Standard-Errors-Confidence-Intervals-and-Other-Measures/10.1214/ss/1177013815.full
    
    """

    def __init__(self, 
            nTrain:int, 
            nTest:int,
            sample_size:int, 
            categoryField:str='index', 
            ):
        assert nTrain>0 and nTest>0, f"nTrain and nTest must be greater than cero"

        self.sample_size = sample_size

        self.splitter = datasetSplit({'train':nTrain,'test':nTest}, categoryField=categoryField)

    def __len__(self): 
        return self.sample_size

    def __call__(self, 
            dataset:pd.DataFrame
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        for _ in range(self.sample_size):
            data = self.splitter(dataset) #splits the dataset into K-folds
            train_dataset = data['train']
            test_dataset = data['test']
            yield train_dataset, test_dataset
        

if __name__ == '__main__':
    import os
    os.system('clear')
    test=1

    if test == 1:   
        print(f'test {test}\nsplit in train:1, test:1')
        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J'],
                            'category':  [1,2,1,1,2,2,3,3,3,1]}) #3 samples of each group

        print(f"data:\n{data}\n{'='*50}")

        splitter = bootStrap(2,categoryField='category')

        for train, test in splitter(data):
            print(f"|train|={len(train)}:\n{train}")
            print(f"|test|={len(test)}:\n{test}\n{'='*50}")
    
