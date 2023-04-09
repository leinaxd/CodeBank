from ..torchModule import torchModule
import torch.nn.functional as F
from torchtyping import TensorType
from itertools import tee#, pairwise python3.10
import torch.nn as nn


def pairwise(iterable):
    a,b = tee(iterable)
    next(b)
    return zip(a,b)

import torch
class multilayerPerceptron(torchModule):
    def __paper__(self): return 'https://www.researchgate.net/publication/228340819_Multilayer_perceptron_and_neural_networks'
    def __init__(self, layer_sizes, bias, sigma):
        super().__init__()
        self.sigma = sigma
        self.layers = nn.ModuleList(
            [nn.Linear(inSize, outSize, bias) for inSize, outSize in pairwise(layer_sizes)]
            )

    def forward(self, x:TensorType['batch_size', 1]):
        out=x
        for layer in self.layers[:-1]:
            out = layer(out)
            out = self.sigma(out)
        out = self.layers[-1](out)
        return out
        
    def loss(self, trainBatch):
        src,tgt = torch.tensor([trainBatch['src']]).T, torch.tensor([trainBatch['tgt']]).T
        prediction = self.forward(src)
        return (tgt-prediction)**2

    def predict(self, trainBatch):
        src = torch.tensor([trainBatch['src']]).T
        return self.forward(src)


class FFNN(nn.Module):
    """Implements FFNN equation.
    FFNN(x) = max(0, xW1 + b1)W2 + b2
    W1: d_model -> d_ff
    W2: d_ff -> d_model
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.sigma = F.relu
    def forward(self, x):
        out = self.w_1(x)
        out = self.sigma(out)
        out = self.dropout(out)
        out = self.w_2(out)
        return out