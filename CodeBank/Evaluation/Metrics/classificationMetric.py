"""
Author: Eichenbaum, Daniel
eMail: eichenbaum.daniel@gmail.com
Last upDate: 9/12/2022
"""
from typing import Union, List, Tuple
from collections import namedtuple
import numpy as np

class classificationMetric:
    """
    classificationMetric

    Given a binary classification task, 
        apply diferrent metrics or save results

    related: confusion Matrix, ROC, AUC

    __init__
        <labels> names of each hypothesis class
    __call__(
        <predictions> the predicted label class ix
        <true_label>  the true label for each prediction
        )
        append a new observation to the current experiment
    close()
        Closes the current experiment and start a new one

    doConfusion(
        <H1_region>         
            Threshold of prob_1 where H1 is true
        <groupByExperiments> compute the confusion matrix in stem or do individual computations
        )

    doROC(
        <Tstart> The threshold start
        <Tend>   The threshold end
        <nSteps> The number of thresholds
        <groupByExperiments> Whereas to Compute diferents ROC for each experiment, or stem all experiments together
        )-> 
        returns: <H1 thresholds>, List[<confusion Matrix>]
            list of confusion matrix computed for diferents thresholds

    Computes
        TP: True positives; count(Predicted(positive) == True(positive))
        TN: True negatives: count(Predicted(negative) == True(negative) )
        FP: False positives: count(Predicted(positive) == True(negative))
        FN: False negative: count(Predicted(negative) == True(positive))
        Accuracy = TP+TN / Total
        ROC: Apply diferent Biases in order to plot the ROC curve
        AUC: Area under the curve
    NOTE.1:
        For true_label, let 0: denote Hypothesis 0 (H0) is true. Alike 1:H1, 2:H2 ...
    NOTE.2: 
        in sklearn the order of meassure is (TN, FP, FN, TP)
            sklearn.metrics.confusion_matrix
        while here the order of meassure is (TP, TN, FP, FN)
    NOTE.3:
        Since, Sum(Probs_i) = 1 there always one redundant probability named H0.
        That's why prob_0 in never requested
    TODO:
        H0_region: could be a list of intervals where H0 holds true for classification
    """
    def __init__(self):
        self.history = [] #list of experiments [[prob_1,true_label] ...]
        self.currentExperiment = [] #list of tuples [(prob_1, prob_2..., true_label) ...]

    def load_state(self, history:list):
        self.history = history
    def save_state(self) -> list:
        return self.history
    def __str__(self):
        return str(self.history)
    def reset(self):
        self.history = []
        self.currentExperiment = []

    def __call__(self, 
        prob_1:Union[list, int],
        true_label:Union[list, int]):
        # prediction = torch.sum(probs,0) #en la ventana sumo las probs
        try: len_prob_1 = sum(1 for _ in prob_1)
        except TypeError:   
            prob_1 = [prob_1]
            len_prob_1 = 1

        try: len_true   = sum(1 for _ in true_label)
        except TypeError:
            true_label = [true_label]
            len_true = 1

        assert len_prob_1 == len_true, f"predicted and true_label missmatched in size"
        self.currentExperiment += [(p,t) for p,t in zip(prob_1, true_label)]

    def close(self):
        """Close the current experiment and starts the next one"""
        if len(self.currentExperiment): 
            self.history.append(self.currentExperiment)
            self.currentExperiment = []

    def compute(self, 
        roc_curve:Union[Tuple[int,int,int,int],List[Tuple[int,int,int,int]]], 
        metrics:  Union[str, List[str]]
        )-> List[List[float]]:
        """
        Given a Sequence of confusion Matrix it return the given metrics
        <types>:
            ACC: Accuracy
            FPR: False Positive Rate
            TPR: True Positive Rate
        )-> metric_1(roc_curve) , metric_2(roc_curve)
            returns the sequence of metrics
        """
        if len(roc_curve) != 1: roc_curve = [roc_curve]
        if isinstance(type, str): metrics = [metrics]
        print(roc_curve)
        return [[self._compute(*confusion, metric) for confusion in roc_curve] for metric in metrics]


    def _compute(self, TP,TN,FP,FN, metric:str):
        if metric == 'TPR': return TP/(TP+FN) if TP+FN else 0
        if metric == 'FPR': return FP/(FP+TN) if FP+TN else 0
        if metric == 'ACC': return (TP+TN)/(TP+TN+FP+FN) if TP+TN+FP+FN else 0



    def doROC(self, nSteps:int=10, Tstart:int=0, Tend:int=1, 
        groupByExperiments=True
        ):
        self.close()
        step = (Tend-Tstart)/nSteps
        thresholds = np.arange(Tstart, Tend+step, step)
        if groupByExperiments:
            result = [ [self._doConfusion(exp, th) for th in thresholds] for exp in self.history]
        else: 
            exp = self._flatten()
            result = [self._doConfusion(exp, th) for th in thresholds]
        return thresholds, result

    def doConfusion(self, 
            H1_region:float=0.5,
            groupByExperiments=True
            ):
        if groupByExperiments: return [self._doConfusion(exp, H1_region) for exp in self.history]
        exp = self._flatten()

        TP,TN,FP,FN = self._doConfusion(exp, H1_region)
        return namedtuple('confusion',('TP','TN','FP','FN'))(TP,TN,FP,FN)


    def _flatten(self):
        """ignore the experiments, appends all results in same list"""
        return [e for exp in self.history for e in exp]

    def _doConfusion(self, experiments:List[Tuple[tuple, int]], H1_region:float):
        TP = TN = FP = FN = 0 #current values
        for prob_1, true_label in experiments:
            #apply the threshold classification
            pred = 1 if H1_region <= prob_1 else 0
            true = int(true_label)

            #be careful, 0 means H0, 1 means H1, 
            # don't confuse with True/False or Positive/Negative
            if pred == 0 and true == 0: #H0 ^ H0
                TP += 1
            if pred == 1 and true == 1: #H1 ^ H1
                TN += 1 
            if pred == 0 and true == 1: #H0 ^ H1
                FP += 1 
            if pred == 1 and true == 0: #H1 ^ H0
                FN += 1 
        return (TP,TN,FP,FN)

    def doAUC(self, x:List[float], y:List[float],sorted=True):#data:List[Tuple[int,int]]):
        """<sorted> is required to avoid negative areas"""
        x, y = np.array(x), np.array(y)
        if sorted: 
            ix = np.argsort(x)
            x = x[ix]
            y = y[ix]
        return np.trapz(y,x) #AUC



if __name__ == '__main__':
    import torch
    import numpy as np
    test = 2
    if test == 1:
        print(f'test {test}: classify for confusion Matrix')
        experiments = [0.68, 0.5, 0.3, 0.9, 0.1] #0 = H0, 1=H1
        pred        = [   1,   1,   0,   1,   0] #0 = H0, 1=H1
        true        = [   0,   0,   0,   1,   0]
        expected    = [2,1,0,2]
        metric = classificationMetric()
        metric(prob_1=experiments, true_label=true)
        metric.close()
        obtained = metric.doConfusion(H1_region=0.5,groupByExperiments=False)
        print(f"experiments:\t{experiments}\npredicted:\t{pred}\ntrue_label:\t{true}\n{'='*50}\nobtained:\t{obtained}\nexpected:\t{expected}")
        print(f"history:\n\t{metric.compute()}")

    if test == 2:
        print(f'test {test}: classify for ROC')
        experiments = [0.68, 0.4, 0.3, 0.9, 0.1, 0.6, 0.7, 0.3, 0.2, 0.16, 0.47,0.53,0.18,0.87] #0 = H0, 1=H1
        experiments *= 1
        pred        = [   1,   0,   0,   1,   0,   1,   1,   0,   0,    0,    0,   1,   0,   1]*1

        true = np.random.randint(0,2,len(experiments))
        # true = pred

        expected    = [0,1,2,2]
        metric = classificationMetric()
        metric(prob_1=experiments, true_label=true)
        print(f"exp:{experiments}\ntrue:{true}")
        th, roc_curve = metric.doROC(nSteps=10, groupByExperiments=False)
        x, y = metric.compute(roc_curve, ['FPR','TPR'])
        # x = []
        # y = []
        # for (TP,TN,FP,FN) in roc_curve:
            # TPR = TP/(TP+FN) if TP+FN else 0
            # FPR = FP/(FP+TN) if FP+TN else 0
            # x.append( FPR )
            # y.append( TPR )
            # print([TP,TN,FP,FN], TPR, FPR)
        ix = np.argsort(np.array(x))
        x = np.array(x)[ix]
        y = np.array(y)[ix]
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        plt.plot(x,y,'o-')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        print(f"Area under curve={metric.doAUC(x,y)}")
        plt.show()


    if test == 3:
        print(f'test {test}: load the metric')
        metric = classificationMetric()
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
        metric = classificationMetric(['H0','H1'])
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
        metric = classificationMetric(['H0','H1'])
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