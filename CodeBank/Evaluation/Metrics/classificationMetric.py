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
        invertHypothesis = True
            H0 = CTR, H1 = disease
            TP: True positive:  count( Reject H0 == True(H1) )
            TN: True negative:  count( Fail Reject H0 == True(H0) )
            FP: False positive: count( Reject H0 == True(H0) )
            FN: False negative:  count( Fail Reject H0 == True(H1))
        invertHypothesis = False
            H0 = disease, H1 = CTR
            TP: True positive: count(Predicted(H0) == True(H0))
            TN: True negative: count(Predicted(H1) == True(H1) )
            FP: False positive: count(Predicted(H0) == True(H1))
            FN: False negative: count(Predicted(H1) == True(H0))
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
    def __init__(self, invertHypothesis=False):
        self.history = [] #list of experiments [[prob_1,true_label] ...]
        self.currentExperiment = [] #list of tuples [(prob_1, prob_2..., true_label) ...]
        assert not invertHypothesis, f"verify invertHypothesis. Its not just inverting hypothesis..."
        self.invertedHypothesis = invertHypothesis

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
        metrics:  Union[str, List[str]],
        )-> List[List[float]]:
        """
        Given a Sequence of confusion Matrix it computes the requested metrics
        Notation:
            hx: predicted 
            Hx: True
        <metrics>:
            P:   TP/(TP+TN+FP+FN)              Prevalence                                       P(h0)
            ACC: (TP+TN)/(TP+TN+FP+FN)         Accuracy                                         P(hx=Hx)
            BA:  (TPR+TNR)/2                   Balanced Accuracy 
            F1:  [0.5*([TPR^-1]+[PPV^-1])]^-1  F1 score
            TPR: TP/(TP+FN)                    True Positive Rate (sensitivity, recall)         P(h0|H0)
            TNR: TN/(TN+FP)                    True Negative Rate (specificity, selectivity)    P(h1|H1)
            PPV: TP/(TP+FP)                    Positive Predicted Value (precision)             P(H0|h0)
            NPV: TN/(TN+FN)                    Negative Predicted Value                         P(H1|h1)
            FNR: FN/(TP+FN)                    False Negative Rate (miss rate, Error I*)        P(h1|H0)
            FPR: FP/(TN+FP)                    False Positive Rate (fall out, Error II*)        P(h0|H1)
            FDR: FP/(TP+FP)                    False Discovery Rate                             P(H1|h0)
            FOR: FN/(TN+FN)                    False Omission Rate                              P(H0|h1)
            LR+: TPR/FPR                       Positive Likelihood Ratio                        P(h0|H0)/P(h0|H1)
            LR-: FNR/TNR                       Negative Likelihood Ratio                        P(h1|H0)/P(h1|H1)
            TS:  TP/(TP+FN+FP)                 Threat Score (critical success index [CSI])      P(h0=H0|h1!=H1)
            PT:  sqrt(FPR)/sqrt(TPR)+sqrt(FPR) Prevalence Threshold
            *warning, if the H0 is reject H0, then the Type Errors swaps.
        <sort>:
            usefull for plotting or for computing AUC,
            however it miss shift the thresholds values
        )-> metric_1(roc_curve) , metric_2(roc_curve)
            returns the sequence of metrics
        """
        if isinstance(metrics, str): metrics = [metrics]
        if isinstance(roc_curve[0], tuple): #stem
            return [[self._compute(*confusion, metric) for confusion in roc_curve] for metric in metrics]
        #groupByExperiments
        return [[[self._compute(*confusion, metric) for confusion in exp] for metric in metrics] for exp in roc_curve]

    def _compute(self, TP,TN,FP,FN, metric:str):
        metric = metric.upper()
        if metric == 'P':   return TP/(TP+TN+FP+FN)      if TP+TN+FP+FN else 0
        if metric == 'ACC': return (TP+TN)/(TP+TN+FP+FN) if TP+TN+FP+FN else 0
        if metric == 'BA':  return (self._compute(TP,TN,FP,FN,'TPR')+self._compute(TP,TN,FP,FN,'TNR'))/2
        if metric == 'F1':  return 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
        if metric == 'TPR': return TP/(TP+FN) if TP+FN else 0
        if metric == 'TNR': return TN/(TN+FP) if TN+FP else 0
        if metric == 'PPV': return TP/(TP+FP) if TP+FP else 0
        if metric == 'NPV': return TN/(TN+FN) if TN+FN else 0
        if metric == 'FNR': return FN/(TP+FN) if TP+FN else 0
        if metric == 'FPR': return FP/(TN+FP) if TN+FP else 0
        if metric == 'FDR': return FP/(TP+FP) if TP+FP else 0
        if metric == 'FOR': return FN/(TN+FN) if TN+FN else 0
        if metric == 'LR+': return self._compute(TP,TN,FP,FN,'TPR')/self._compute(TP,TN,FP,FN,'FPR')
        if metric == 'LR-': return self._compute(TP,TN,FP,FN,'FNR')/self._compute(TP,TN,FP,FN,'TNR')
        if metric == 'TS':  return TP/(TP+FN+FP) if TP+FN+FP else 0
        if metric == 'PT':  
            num = np.sqrt(self._compute(TP,TN,FP,FN,'FPR'))
            den = num + np.sqrt(self._compute(TP,TN,FP,FN,'TPR'))
            return num/den if den else 0
        



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
            if self.invertedHypothesis:
                if pred == 1 and true == 1: #Reject H0 ^ H1
                    TP += 1 
                if pred == 0 and true == 0: #Fail Reject H0 ^ H0
                    TN += 1
                if pred == 0 and true == 1: #Fail Reject H0 ^ H1
                    FN += 1 
                if pred == 1 and true == 0: #Reject H0 ^ H0
                    FP += 1 
            else:
                if pred == 0 and true == 0: #H0 ^ H0
                    TP += 1
                if pred == 1 and true == 1: #H1 ^ H1
                    TN += 1 
                if pred == 0 and true == 1: #H0 ^ H1
                    FP += 1 
                if pred == 1 and true == 0: #H1 ^ H0
                    FN += 1 
        return (TP,TN,FP,FN)

    def doAUC(self, x:List[float], y:List[float], thresholds:List[float], sorted=True):#data:List[Tuple[int,int]]):
        """<sorted> is required to avoid computing negative areas
        )-> AUC, sorted[x],sorted[y]
        return:
            The <AUC> score
            <x>, <y>, <thresholds> is useful for plotting
        """
        x = np.array(x) if isinstance(x, list) else x
        y = np.array(y) if isinstance(y, list) else y
        th = np.array(thresholds) if isinstance(thresholds, list) else thresholds
        if sorted: 
            ix = np.argsort(x)
            x =  x[ix]
            y =  y[ix]
            th = th[ix]
        return np.trapz(y,x), x, y, th #AUC



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
        experiments *= 1000
        true        = [   1,   0,   0,   1,   0,   1,   1,   0,   0,    0,    0,   1,   0,   1]*1

        true = np.random.randint(0,2,len(experiments))

        expected    = [0,1,2,2]
        metric = classificationMetric()
        metric(prob_1=experiments, true_label=true)
        th, roc_curve = metric.doROC(nSteps=10, groupByExperiments=False)

        #Inverted hypothesys
        # y_label, x_label = 'ACC', 'FPR' #Acc vs EI
        # y_label, x_label = 'TPR', 'FPR' #P(h0|H0) vs P(h0|H1) *para rechazar hipótesis
        # y_label, x_label = 'FPR', 'FNR' #EI vs EII

        #noninverted Hypothesys
        # y_label, x_label = 'ACC', 'FNR' #Acc vs EI
        y_label, x_label = 'TNR', 'FNR' #P(h1|H1) vs P(h1|H0)
        # y_label, x_label = 'FNR', 'FPR' #EI vs EII

        x, y = metric.compute(roc_curve, [x_label,y_label])
        # x, y = metric.compute(roc_curve, ['FPR','TPR'])
        AUC, x, y, th = metric.doAUC(x,y,th)
        if len(experiments)<20: print(f"exp:{experiments}\ntrue:{true}\nth:{th}")
        print(f"Area under curve={AUC}")
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        plt.plot(x,y,'o-')
        for _x,_y,_th in zip(x,y,th): plt.text(_x,_y,f"{_th:.2}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        alpha = 0.05
        plt.axvline(x=alpha)
        plt.text(x=alpha+0.01,y=0.5,s=f"$\\alpha$={alpha}")
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