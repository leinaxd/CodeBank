
import torch, torch.nn as nn
from torchtyping import TensorType
class old_labelSmoothing(nn.Module):
    """Implement label smoothing. with Kullback Leibler loss function

    We create a distribution that has confidence on the correct word,
    and the rest of mass is smoothed throughout the vocabulary.

    Label Smoothing penalizes the model if it gets very confident about a given choise
    Parameters
        <confidence>: Amount of probability to difuse
    Input
        tgt_ix: The true answers indexes
    Output
        A smoothed true answer distribution
    Method
        Higher value = <confident>
        Lower value = (1-<smoothing>)/<seq_len>
    Paper? 
        https://arxiv.org/abs/1512.00567
    """
    def __init__(self, padding_idx, confidence=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss()
        # self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = confidence
        self.smoothing = 1.0-confidence

    def forward(self,   prediction: TensorType['batch_size','seq_len','output_size'], 
                        tgt_ix    : TensorType['batch_size','seq_len','output_size']
                        ) ->        TensorType['batch_size']:
        seq_len = prediction.size(1)
        #create a smooth Window
        true_dist = torch.full_like(prediction, self.smoothing/(seq_len-2))
        # Define lower value: self.smoothing/(seq_len-2)
        # Define high value: self.confidence
        # true_dist.fill_(self.smoothing / (seq_len - 2))

        # Por columnas
        true_dist.scatter_(1, tgt_ix.unsqueeze(1), self.confidence)
        



        return self.criterion(prediction, true_dist), true_dist
