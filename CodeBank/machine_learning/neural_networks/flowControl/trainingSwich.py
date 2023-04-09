import torch.nn as nn

class trainingSwitch(nn.Module):
    """
    Selects diferents models for evaluation and training

    >>> model = trainingSwitch(evalModel, trainModel)
    >>> model.foward() == trainModel.forward() #True if model.train == True
    >>> model.foward() == evalModel.forward()  #True if model.eval == True
    """
    def __init__(self, evalModel, trainModel):
        super().__init__()
        self.evalModel = evalModel
        self.trainModel = trainModel
    def forward(self, *args, **kwargs):
        if self.training: return self.trainModel.forward(*args, **kwargs)
        else:             return self.evalModel.forward(*args, **kwargs)