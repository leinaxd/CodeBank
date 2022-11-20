from typing import Union, Sequence
from collections import namedtuple

class confusionMatrix:
    """
    Confusion matrix

    Given a binary classification task, 

    __init__
        <labels> names of each hypothesis class


    __call__
        <predictions> the predicted label class ix
        <true_label>  the true label for each prediction

    Computes
        TP: True positives; count(Predicted(positive) == True(positive))
        TN: True negatives: count(Predicted(negative) == True(negative) )
        FP: False positives: count(Predicted(positive) == True(negative))
        FN: False negative: count(Predicted(negative) == True(positive))
        Accuracy = TP+TN / Total

    NOTE: 
        in sklearn the order of meassure is (TN, FP, FN, TP)
            sklearn.metrics.confusion_matrix
        while here the order of meassure is (TP, TN, FP, FN)
        
    """
    def __init__(self, labels:Sequence[str]):
        self.history = [] #list of tuples (TP,TN,FP,FN)
        self.labels = labels
        self.TP = self.TN = self.FP = self.FN = 0 #current values


    def load_state(self, history:list):
        self.history = history
    def save_state(self) -> list:
        return self.history

    def __str__(self):
        return str(self.history)

    def reset(self):
        self.history = []

    def __call__(self, 
        predicted:Union[list,int], 
        true_label:Union[list,int]):
        #TODO:
        # if predicted/true_label are a batch of tensors? they do not cast to a list...
        if isinstance(predicted, int):  predicted = [predicted]
        if isinstance(true_label, int): true_label = [true_label]
        # if isinstance(true_label)
        len_predicted = sum(1 for _ in predicted)
        len_true      = sum(1 for _ in true_label)
        assert len_predicted == len_true, f"predicted and true_label missmatched in size"

        #be careful, positive means H0 and negative means H1, 
        # don't confuse with boolean True/False
        for pred, true in zip(predicted,true_label):
            if int(pred) == 0 and int(true) == 0: #H0 ^ H0
                self.TP += 1
            if int(pred) == 1 and int(true) == 1: #H1 ^ H1
                self.TN += 1 
            if int(pred) == 0 and int(true) == 1: #H0 ^ H1
                self.FP += 1 
            if int(pred) == 1 and int(true) == 0: #H1 ^ H0
                self.FN += 1 
            

    def compute(self, type:str=''):
        #update the batch
        if self.TP or self.TN or self.FP or self.FN: #if new data
            self.history.append((self.TP, self.TN, self.FP, self.FN))
            self.TP = self.TN = self.FP = self.FN = 0

        #compute the metric
        if type.lower() in [None,'']:
            return self.history
        elif type.lower() in ['confusion','c']:
            return self.doConfusion()
        elif type.lower() in ['acc','accuracy']:
            return self.doAccuracy()
        else:
            raise NotImplementedError(f"{type} is not implemented yet")

    def doAccuracy(self):
        return [(TP+TN)/(TP+TN+FP+FN) for TP,TN,FP,FN in self.history]
    def doConfusion(self):
        class confusion:
            def __init__(self):
                self.TP,self.TN,self.FP,self.FN = 0,0,0,0
            def get_item(self, key:str):
                if key=='TP': return self.TP
                if key=='TN': return self.TN
                if key=='FP': return self.FP
                if key=='FN': return self.FN
            def __str__(self):
                return f"TP={self.TP}, TN={self.TN}, FP={self.FP}, FN={self.FN}"
        result = confusion()
        for TP,TN,FP,FN in self.history:
            result.TP += TP
            result.TN += TN
            result.FP += FP
            result.FN += FN
        return result

if __name__ == '__main__':
    test = 1
    if test == 1:
        print(f'test {test}: load the metric')
        metric = confusionMatrix(['H0','H1'])
        print(f'(TP,TN,FP,FN)')
        for true, predicted, expected in zip(
            [[0,0,0,0,1,1,1,1],
             [0,0,0,0,1,1,1,1],
             [0,0,0,0,1,1,1,1],
             [0,0,0,0,1,1,1,1]],

            [[0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1],
             [0,0,1,1,0,0,1,1],
             [1,1,1,1,0,0,0,0]],

            [[4,0,4,0],
             [0,4,0,4],
             [2,2,2,2],
             [0,0,4,4]]):
            metric(predicted,true)
            metric.compute()
            print(f"true:{true}\t|predicted:{predicted}\t|expected:{expected}")
        print('history:',metric.compute())
        print(metric.compute('acc'))
        print(metric.compute('confusion'))
    if test == 2:
        import torch
        import numpy as np
        print(f'test {test}: numpy/torch compatibility')
        metric = confusionMatrix(['H0','H1'])
        print(f'(TP,TN,FP,FN)')
        for true, predicted, expected in zip(
            [np.array([0,0,0,0,1,1,1,1]),
             torch.tensor([0,0,0,0,1,1,1,1])],

            [[0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1]],

            [[4,0,4,0],
             [0,4,0,4]]):
            metric(predicted,true)
            metric.compute()
            print(f"true:{true}\t|predicted:{predicted}\t|expected:{expected}")
        print('history:',metric.compute())
        print(metric.compute('acc'))
        print(metric.compute('confusion'))
