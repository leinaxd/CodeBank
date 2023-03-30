import torch.nn as nn

class goldenDecoder(nn.Module):
    """
    Trains the decoder with true golden targets for each sequence step
    """
    def __init__(self, decoderModule):
        super().__init__()
        self.decoderModule = decoderModule
    def forward(self, *args, **kwargs):
        return self.decoderModule(*args, **kwargs)