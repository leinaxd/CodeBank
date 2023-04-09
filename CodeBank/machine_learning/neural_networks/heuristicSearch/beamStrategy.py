from torchtyping import TensorType
import torch, torch.nn as nn

class beamStrategy(nn.Module):
    """
    The heuristic of selecting at heach step the K most probable secuences.
    """
    def __init__(self, beam_size:int):
        super().__init__()
        self.beam_size = beam_size
        
    def forward(self, dec_out:TensorType['batch_size', 'prob_len']
                      )  ->   TensorType['batch_size']:
        symbols = torch.topk(dec_out, self.beam_size, -1)
        return symbols


