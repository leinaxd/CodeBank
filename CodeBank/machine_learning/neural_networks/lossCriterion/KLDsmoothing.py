
import torch, torch.nn as nn
from torchtyping import TensorType
from CodeBank.NeuralNetworks.masking import labelSmoothing

class KLDsmoothing(nn.Module):
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
    Papers
        - Rethinking the Inception Architecture for Computer Vision
            https://arxiv.org/abs/1512.00567


    Related:
        nn.CrossEntropyLoss()
    """
    def __init__(self, confidence=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        # self.criterion = nn.KLDivLoss(size_average=False)
        self.lossMask = labelSmoothing(confidence)

    def forward(self,   prediction: TensorType['batch_size','seq_len','output_size'], 
                        tgt_ix    : TensorType['batch_size','seq_len','output_size']
                        ) ->        TensorType['batch_size']:

        true_dist = self.lossMask(prediction, tgt_ix)
        return self.criterion(prediction, true_dist)
