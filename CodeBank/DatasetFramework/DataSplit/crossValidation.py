import pandas as pd
import itertools


try:
    from datasetSplit import datasetSplit
except:
    from . import datasetSplit

class crossValidation:
    """
    Performs the cross validation loop

    1. Splits the dataset into K-folds
        parameter k number of folds
        parameter test_size number of test groups combined for test (the rest goes into the training)
    2. returns tuples of train-val


    __call__(dataset:pd.DataFrame):
        yield train_dataset, test_dataset


    Sources:
        - https://www.wikiwand.com/es/Validaci%C3%B3n_cruzada
        - https://arxiv.org/abs/2104.00673
        - Encyclopedia of Database Systems (Liu, Tamer)
    Others
        - https://www.wikiwand.com/en/Binomial_coefficient
    """
    def __init__(self, 
                k_fold:int, 
                test_size:int=1,
                categoryField:str='index',
                verbose=False):
        assert k_fold>test_size, f"you can't take {test_size} groups from a len {k_fold} list"
        self.k_fold = k_fold
        self.test_size=test_size
        self.verbose=verbose
        
        self.foldsLabels = [f"{k+1}-fold" for k in range(self.k_fold)]
        self.splitter = datasetSplit(self.foldsLabels, categoryField=categoryField)
    def labelSelection(self):
        for test_fold in itertools.combinations(self.foldsLabels, self.test_size):
            train_fold = [label for label in self.foldsLabels if label not in test_fold]
            yield train_fold, test_fold

    def __call__(self,dataset:pd.DataFrame):
        assert len(dataset) > self.k_fold, f"You can't split the dataset of len {len(dataset)} into {self.k_fold} parts"
        data = self.splitter(dataset) #splits the dataset into K-folds
        if self.verbose:
            for k,v in data.items():
                print(f"{k}\n{v}\n{'='*50}")


        for train_fold, test_fold in self.labelSelection():
            test_dataset  = pd.concat([data[key] for key in test_fold])
            train_dataset = pd.concat([data[key] for key in train_fold])
            if self.verbose: 
                print(f"test_fold:\t{test_fold}, len: {len(test_dataset)}")
                print(f"train_folds:\t{train_fold}, len: {len(train_dataset)}\n{'='*50}")

            yield train_dataset, test_dataset

if __name__=='__main__':
    import os
    os.system('clear')

    test = 3
    if test == 1:
        print(f"test {test}: fold labels")
        dataset = pd.DataFrame([[-10000,-20000,-30000],[-1000,-2000,-3000],[-100,-200,-300],[1,2,3],[10,20,30],[100,200,300],[1000,2000,3000],[10000,20000,30000]])
        print(f"dataset:\n{dataset}\n{'='*50}")
        splitter = crossValidation(3,2,verbose=True)
        for train_dataset, test_dataset in splitter(dataset):
            print(f"train:\n{train_dataset}")
            print(f"test:\n{test_dataset}")
            

    if test == 2:
        print(f"test {test}: combinatorics")
        labels = ['a', 'b','c','d','e']
        print(labels)
        combinations = itertools.combinations(labels,2)
        test_folds = list(combinations)
        train_folds = [l for l in labels if l not in test_folds[1]]
        
        print(test_folds)
        print(train_folds)
    if test == 3:
        print(f"test {test} labels")
        splitter = crossValidation(6,2)
        for train, test in splitter.labelSelection():
            print(f"train:\t{train}")
            print(f"test:\t{test}")


    