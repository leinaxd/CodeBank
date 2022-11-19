from typing import Union, Sequence

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
        if isinstance(predicted, int):  predicted = [predicted]
        if isinstance(true_label, int): true_label = [true]
        assert len(predicted) == len(true_label), f"predicted and true_label missmatched in size"

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
        if self.TP or self.TN or self.FP or self.FN: #if new data
            self.history.append((self.TP, self.TN, self.FP, self.FN))
            self.TP = self.TN = self.FP = self.FN = 0

        if type.lower() in [None,'']:
            func = lambda TP,TN,FP,FN:(TP,TN,FP,FN) #identity
        elif type.lower() in ['acc','accuracy']:
            func = lambda TP,TN,FP,FN: (TP+TN)/(TP+TN+FP+FN)
        else:
            raise NotImplementedError(f"{type} is not implemented yet")
        
        return [func(TP,TN,FP,FN) for TP,TN,FP,FN in self.history]



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