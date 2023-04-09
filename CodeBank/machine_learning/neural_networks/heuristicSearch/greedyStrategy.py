from torchtyping import TensorType
import torch, torch.nn as nn

    
class greedyStrategy(nn.Module):
    """
    The heuristic of selecting the most probable symbol 
        at each time step of the sequence.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, measure: TensorType['batch_size', 'seq_len', 'prob_len']
                      )  ->    TensorType['batch_size']:
        if len(measure.shape) == 3:
            measure = measure[:,-1,:] #just take the last symbol
        symbols = torch.argmax(measure,-1)
        return symbols



