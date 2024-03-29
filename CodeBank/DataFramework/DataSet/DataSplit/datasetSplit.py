"""
Author: Eichenbaum, Daniel.
Date: 8/11/2022

Used by:
    -bootStrapping
    -crossValidation
"""
import time
from random import Random
from typing import Dict, Union, List
import pandas as pd

class datasetSplit:
    """
    Description:
        Split the data (pandas.DataFrame) into sets (usually 'train', 'test', 'val')
        if the categoryField is set, then it will try to priotize balaced groups of each class
    Parameters:
        groupsProportions: {'train':1, 'val':1, 'test':1}
            a dict containing the proportions for each class
        categoryField: 'index'
            the name of the class header used for creating balaced datasets
        randomSeed: True
            use a random generated seed
    Example:
        >>> data = pd.DataFrame(
                {'data':['A','B','C','D','E', 'F','G','H','I','J'],
                 'category':  [1,2,1,1,2,2,3,3,3,2]} #3 samples of each group
                ) 
        >>> group = 'category'
        >>> proportions = {'train':2, 'val':1, 'test':1}
        >>> splitter = datasetSplit(groupsProportions=proportions,categoryField=group)

        >>> print(f"data\tlen:{len(data)}\n{data}\n\n\n")
        >>> data = splitter(data)
        >>> print(f"train\tlen:{len(data['train'])}\n{data['train']}\n")
        >>> print(f"test\tlen:{len(data['test'])}\n{data['test']}\n")
        >>> print(f"val\tlen:{len(data['val'])}\n{data['val']}\n")
        
    References:
        https://www.v7labs.com/blog/train-validation-test-set
    """
    
    #TODO: ixGenerator should be local, not a class member
    seed = 0
    ixGenerator = Random(seed) #An own generator guarantees reproducibility, otherwise 
                               # an external sample of this module would shift the sequence, 
                               # breaking determinism.

    def __init__(self, 
            groupsProportions:Union[dict, list]={'train':1, 'val':1,'test':1}, 
            categoryField:Union[str,List[str]]='index', 
            prioritySmaller=True, 
            shuffle=True,
            randomSeed=True,
            forceEqualLen=False):
        #identities
        self.categoryField = categoryField
        self.shuffle=shuffle
        self.priority = not prioritySmaller
        self.forceEqualLen=forceEqualLen
        #Default values
        if isinstance(groupsProportions, list): groupsProportions = {k:1 for k in groupsProportions}
        #Normalize proportions
        den = sum(groupsProportions.values())
        self.groupsProportions = {key:value/den for key,value in groupsProportions.items()}
        #random seed
        if randomSeed: self.__class__.ixGenerator = Random(time.time())
    def setSeed(self, seed):
        self.ixGenerator.seed(seed)    
        
    def splitRatio(self, data_size:int) -> dict:
        """
        If there were loose samples, 
            then include them by lower/higher proportion order first
        """
        proportions = {key:int(value*data_size) for key, value in self.groupsProportions.items()}
        rem = data_size-sum(proportions.values())
        sorted_proportions = sorted(self.groupsProportions.items(), key=lambda x:x[1], reverse=self.priority)
        for field, _ in sorted_proportions:
            if rem == 0: break
            proportions[field] += 1
            rem -= 1
        return proportions

    def sampleByIndex(self, data:pd.DataFrame) -> dict:
        groups_len  = self.splitRatio(len(data))

        samples = list(range(len(data)))
        if self.shuffle: self.ixGenerator.shuffle(samples)
        samples_ix = {}
        start = 0
        for key, size in groups_len.items():
            end = start+size
            samples_ix[key] = samples[start:end]
            start = end
        return samples_ix

    def sampleByGroup(self, data:pd.DataFrame) -> Dict[str,list]:
        """
        groups: {key:size}
        return: {key:samples_ix}
        """
        group_sizes = self.splitRatio(len(data))

        key_ix          = 0 #place the new index in this position
        keys = list(group_sizes.keys())

        samples = data.groupby(self.categoryField).indices
        for key, ixs in samples.items(): #{group_class : index_list}
            ixs = ixs.tolist()
            if self.shuffle: self.ixGenerator.shuffle(ixs)
            samples[key] = ixs

        group_class_ix  = 0 
        group_keys = list(samples.keys())
        samples_ix = {key:[] for key in keys}
        while sum(group_sizes.values()):
            # continue the balanced_class 1,2,3 from before
            group_class_ix = (group_class_ix+1) % len(group_keys)
            group_class    =  group_keys[group_class_ix]
            for _ in range(len(group_keys)): 
                #do one loop with same group_class
                if not len(samples[group_class]): break
                ix = samples[group_class].pop() #pop 1 class_ix
                
                while True: 
                    #continue the test, train, val cycle from before
                    key = keys[key_ix]
                    key_ix = (key_ix + 1) % len(keys)
                    if group_sizes[key] == 0: continue #data_size is full cover
                    break

                #Fill one sample,
                group_sizes[key] -= 1 
                samples_ix[key].append(ix)
                # print(key,group_class,ix)
        return samples_ix

    def forcedSampledGroup(self, data:pd.DataFrame):
        
        samples = data.groupby(self.categoryField).indices

        for key, ixs in samples.items(): #{group_class : index_list}
            ixs = ixs.tolist()
            if self.shuffle: self.ixGenerator.shuffle(ixs)
            samples[key] = ixs
        
        if self.forceEqualLen:
            maxLen = min([len(v) for v in samples.values()])
            samples = {k:v[:maxLen] for k,v in samples.items()}
        groups_len = self.splitRatio(maxLen)
        
        samples_ix = {key:[] for key in groups_len.keys()}
        while sum(groups_len.values()):
            for group, count in groups_len.items():
                if not count: continue
                groups_len[group] -= 1
                for v in samples.values(): #paste 1 item of each class
                    samples_ix[group].append(v.pop())

        return samples_ix

    def __call__(self, data:pd.DataFrame, returnSamples=False, resetIx=True) -> pd.DataFrame:
        if self.categoryField in [None, 'index']: samples_ix  = self.sampleByIndex(data)
        elif self.forceEqualLen:                  samples_ix  = self.forcedSampledGroup(data)
        else:                                     samples_ix  = self.sampleByGroup(data)
        
        if returnSamples:  return samples_ix

        sets = {key:data.iloc[ix] for key, ix in samples_ix.items()}
        
        if resetIx: #reset original index
            for k, v in sets.items(): sets[k]=v.reset_index()
        return sets



if __name__ == '__main__':
    def do(data, group='index', proportions={'train':1,'val':1,'test':1},forced=False):
        splitter = datasetSplit(groupsProportions=proportions,categoryField=group,forceEqualLen=forced)
        print(f"{'='*50}\ndata\tlen:{len(data)}\n{data}\n{'='*50}")
        print(f"proportions: {proportions}\n")
        data = splitter(data)
        for key in proportions:
            print(f"{key}\tlen:{len(data[key])}\n{data[key] if len(data[key]) else ''}\n")
        print('\n')

    test = 8
    if test == 1:
        data = pd.DataFrame({'class A':['A'],
                             'class B':['alpha'],
                             'group':  [1],
                             'class C': ['enero']})

        print(f'test {test} a: data of size 1')
        do(data)
        data = pd.DataFrame({'class A':['A','B'],
                             'class B':['alpha','beta'],
                             'group':  [1,1],
                             'class C': ['enero','febrero']})
        print(f'test {test} b: data of size 2')
        do(data)
    if test == 2:
        print(f'test {test}: sample by index')
        data = pd.DataFrame({'class A':['A','B','C','D','E'],
                             'class B':['alpha','beta', 'gamma','delta','epsilon'],
                             'group':  [1,2,2,1,2],
                             'class C': ['enero','febrero','marzo','abril','mayo']})
        do(data)
    elif test == 3:
        print(f'test {test}: sample by group')
        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J'],
                             'category':  [1,2,1,1,2,2,3,3,3,1]}) #3 samples of each group

        do(data,'category', {'train':1,'val':1,'test':1})
    elif test == 4:
        print(f'test {test}\nsplit in train:1, val:2, test:2')
        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J'],
                             'category':  [1,2,1,1,2,2,3,3,3,1]}) #3 samples of each group
        do(data,'category', {'train':1,'val':2,'test':2})
        print('assert: \ntrain: [1,2]\ntest: [1,1,2,3]\nval: [1,2,3,3]')
        print('or:\ntrain: [2,3]\ntest: [2,3,1,1]\nval: [2,3,1,1]')
        print('or:\ntrain: [1,3]\ntest: [1,2,3,1]\nval: [1,2,2,3]')

    elif test == 5:
        print(f'test {test}\nidem test {test-1} but the last class 2 will repeat')
        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J'],
                             'category':  [1,2,1,1,2,2,3,3,3,2]}) #3 samples of each group
        do(data,'category', {'train':1,'val':1,'test':1})

    elif test == 6:
        print(f'test {test}\nforce equal length split')
        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J'],
                             'category':  [1,2,1,1,2,2,3,3,3,2]}) #3 samples of each group
        do(data,'category', {'train':1,'val':1,'test':1},True)

        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J'],
                             'category':  [1,2,1,1,2,2,3,3,3,1]}) #3 samples of each group
        do(data,'category', {'train':1,'val':2,'test':2},True)
    elif test == 7:
        print(f'test {test}\nforce equal length split. Parte 2. More cases')
        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J','k','l','m','n','o','p'],
                             'category':  [1,2,1,1,2,2,3,3,3,1,1,1,1,1,1,1]}) #3 samples of each group
        do(data,'category', {'train':1,'val':2,'test':2},True)

        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J','k','l','m','n','o','p'],
                             'category':  [1,2,1,1,2,2,3,3,3,1,1,1,1,1,2,3]}) #3 samples of each group
        do(data,'category', {'train':1,'val':2,'test':2},True)


        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J','k','l','m','n','o','p','q','r','s'],
                             'category':  [1,2,1,1,2,2,3,3,3,1,1,1,1,1,2,3,1,2,3]}) #3 samples of each group
        do(data,'category', {'train':1,'val':2,'test':2},True)

        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J','k','l','m','n','o','p','q','r','s','t','u','v'],
                             'category':  [1,2,1,1,2,2,3,3,3,1,1,1,1,1,2,3,1,2,3,1,2,3]}) #3 samples of each group
        do(data,'category', {'train':1,'val':2,'test':2},True)
    elif test == 8:
        print(f'test {test}\nforce equal length within 2 categories')
        data = pd.DataFrame({'data':['A','B','C','D','E', 'F','G','H','I','J','k','l','m','n','o','p'],
                             'category_1':  [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3],
                             'category_2':  [1,1,1,2,2,1,1,1,2,2,1,1,1,1,2,2]}) #3 samples of each group
        do(data,['category_1', 'category_2'], {'train':1,'test':1},True)