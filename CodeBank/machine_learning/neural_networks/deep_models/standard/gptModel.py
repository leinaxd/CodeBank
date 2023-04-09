import torch.nn as nn
class gptModel (nn.Module): 
    def __init__(self, input_size, embed_size):
        super().__init__()
