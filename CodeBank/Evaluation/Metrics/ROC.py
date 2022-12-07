#TODO:
# DEPRECATED, USE CONFUSIONMATRIX INSTEAD
from typing import Union, Sequence, List
from collections import namedtuple

class ROC:
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
        ROC: Apply diferent Biases in order to plot the ROC curve
        AUC: Area under the curve
    NOTE: 
        in sklearn the order of meassure is (TN, FP, FN, TP)
            sklearn.metrics.confusion_matrix
        while here the order of meassure is (TP, TN, FP, FN)
    TODO:
        H0_region: could be a list of intervals where H0 holds true for classification
    """
    def __init__(self, H0_region_start:int=0, H0_region_end:int=1, nSteps:int=10):
        self.history = [] #list of H0 tuples [(prob_0, true_label), ...]
        self.H0_region = (H0_region_start, H0_region_end, nSteps)
    def load_state(self, history:list):
        self.history = history
    def save_state(self) -> list:
        return self.history
    def __str__(self):
        return str(self.history)
    def reset(self):
        self.history = []

    def __call__(self, 
        prob_0:Union[list, int],
        true_label:Union[list, int]):
        # prediction = torch.sum(probs,0) #en la ventana sumo las probs
        try: len_prob_0 = sum(1 for _ in prob_0)
        except TypeError:   
            prob_0 = [prob_0]
            len_prob_0 = 1

        try: len_true   = sum(1 for _ in true_label)
        except TypeError:
            true_label = [true_label]
            len_true = 1

        assert len_prob_0 == len_true, f"predicted and true_label missmatched in size"
        self.history += [(p,t) for p,t in zip(prob_0, true_label)]


            
    def compute(self, type:str='',*args):
        #compute the metric
        if type.lower() in [None,'']:
            return self.history
        elif type.lower() in ['roc','r']:
            return self.doROC()
        elif type.lower() in ['confusion','c']:
            return self.doConfusion(*args)
        elif type.lower() in ['auc','a']:
            return self.doAUC()
        else:
            raise NotImplementedError(f"{type} is not implemented yet")

    def doConfusion(self, H0_region):
        TP = TN = FP = FN = 0 #current values
        for prob_0, true_label in self.history:
            #apply the threshold classification
            pred = 0 if H0_region <= prob_0 else 1
            true = int(true_label)

            #be careful, positive means H0 and negative means H1, 
            # don't confuse with boolean True/False
            if pred == 0 and true == 0: #H0 ^ H0
                TP += 1
            if pred == 1 and true == 1: #H1 ^ H1
                TN += 1 
            if pred == 0 and true == 1: #H0 ^ H1
                FP += 1 
            if pred == 1 and true == 0: #H1 ^ H0
                FN += 1 
        return namedtuple('confusion',('TP','TN','FP','FN'))(TP,TN,FP,FN)

    def doAUC(self):
        # TP,TN,FP,FN = self.doConfusion()
        # return [(TP+TN)/(TP+TN+FP+FN) for TP,TN,FP,FN in self.history]
        raise NotImplementedError
    def doROC(self):
        result = []
        thresholds = np.arange(*self.H0_region)
        for th in thresholds:
            result.append(self.doConfusion(th))
        return thresholds, result


if __name__ == '__main__':
    import torch
    import numpy as np
    test = 1
    if test == 1:
        print(f'test {test}: classify for ROC')
        threshold = 0.5
        experiments = [0.68, 0.4, 0.3, 0.9, 0.1] #0 = H0, 1=H1
        pred        = [   0,   1,   1,   0,   1] #0 = H0, 1=H1
        true        = [   1,   1,   0,   1,   0]
        # experiments > 0.5 #predicted = 0, true = 1
        expected    = [0,1,2,2]
        metric = ROC(0,1,10)
        metric(prob_0=experiments, true_label=true)
        confusionMatrix = metric.compute('c',threshold)
        print(f"experiments:\t{experiments}\npredicted:\t{pred}\ntrue_label:\t{true}\n{'='*50}\nobtained:\t{confusionMatrix}\nexpected:\t{expected}")
        print(f"history:\n\t{metric.compute()}")
    if test == 2:
        print(f'test {test}: load the metric')
        metric = ROC()
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
    if test == 3:
        print(f'test {test}: numpy/torch compatibility')
        metric = ROC(['H0','H1'])
        print(f'(TP,TN,FP,FN)')
        for true, predicted, expected in zip(
            [np.array([0,0,0,0,1,1,1,1]),
             torch.tensor([0,0,0,0,1,1,1,1]),
             np.array(1),
             torch.tensor(0)],

            [[0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1],
             0,
             0],

            [[4,0,4,0],
             [0,4,0,4],
             [0,0,1,0],
             [1,0,0,0]]):
            metric(predicted,true)
            metric.compute()
            print(f"true:{true}|predicted:{predicted}|expected:{expected}={metric.compute()[-1]}")
        print('history:',metric.compute())
        print(metric.compute('acc'))
        print(metric.compute('confusion'))
    if test == 4:
        print(f'test {test}: confusion matrix')
        metric = ROC(['H0','H1'])
        print(f'(TP,TN,FP,FN)')
        for true, predicted, expected in zip(
            [[0,0,0,0,1,1,1,1],
             [0,0,0,0,1,1,1,1]],

            [[0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1]],

            [[4,0,4,0],
             [0,4,0,4]]):
            metric(predicted,true)
            metric.compute()
            print(f"true:{true}\t|predicted:{predicted}\t|expected:{expected}")
        print('history:',metric.compute())
        TP, TN, FP, TN = metric.compute('confusion')
        print((TP,TN,FP,TN))
        print( metric.compute('confusion') )