
import torch, torch.nn as nn
from torchtyping import TensorType

class loglikelihood(nn.Module):
    """
    Input
        Prediction holds the predicted distribution
        Tgt is the true observation
    Output:
        loglikelihood = Prediction(obs)
    Note:
        This is not the true loglikehood as we don't have the true PDF

    """
    def __init__(self):
        super().__init__()
    def forward(self,   prediction: TensorType['batch_size', 'seq_len', 'output_size'], 
                        tgt:        TensorType['batch_size', 'seq_len', 'output_size']
                        )       ->  TensorType['batch_size', 'seq_len']:
        device = prediction.device
        if prediction.shape[1] < tgt.shape[1]:
            zeros = torch.zeros([prediction.shape[0], tgt.shape[1]-prediction.shape[1], prediction.shape[2]], device=device)
            prediction = torch.cat((prediction, zeros),dim=1)
        loglikelihood  = torch.gather(prediction, index=tgt.unsqueeze(-1), dim=-1).squeeze(-1)

        return loglikelihood



if __name__ == '__main__':
    pass