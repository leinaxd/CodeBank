from typing import Tuple, Union, List
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
        <train_test_pcn> proportion of train samples vs test samples. 
            train_test_pcn=1, means no test dataset
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
            train_test_pcn:float, 
            sample_size:int, 
            categoryField:Union[str,List[str]]='index', 
            forceEqualLen=False,
            ):
        assert 0<train_test_pcn and train_test_pcn < 1, f"train_test_pcn must lay between (0,1)"
        
        self.sample_size = sample_size
        nTrain = train_test_pcn
        nTest  = 1-train_test_pcn
        self.splitter = datasetSplit({'train':nTrain,'test':nTest}, categoryField=categoryField, forceEqualLen=forceEqualLen)
        self.state = []
    def __len__(self): 
        return self.sample_size

    def loadState(self, state):
        self.state = state
    def generateState(self, dataset:pd.DataFrame):
        """generate the ix for paired data"""
        raise NotImplementedError
        self.state = [] #simulation list of dicts of ix
        for _ in range(self.sample_size):
            self.state.append( self.splitter(dataset, returnSamples=True) )
        return self.state

    def __call__(self, 
            dataset:pd.DataFrame,
            seed=0,
            returnSamples=False
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if seed: self.splitter.setSeed(seed) #make a deterministic choice

        for _ in range(self.sample_size):
            data = self.splitter(dataset, returnSamples=returnSamples)
            yield data['train'], data['test']
        # for samples_ix in state:
            # train_dataset = dataset.iloc[samples_ix['train']].reset_index()
            # test_dataset = dataset.iloc[samples_ix['test']].reset_index()
            # yield train_dataset, test_dataset
        

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
    
