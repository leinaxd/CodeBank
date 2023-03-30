import torch.nn as nn


class logSoftmax(nn.Module):
    # TODO:
    #   delete this module
    "Define standard linear + softmax generation step."
    def __init__(self, embed_size, output_size):
        super().__init__()
        self.proj = nn.Linear(embed_size, output_size)
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        # self.model = nn.Sequential(
        #     nn.Linear(embed_size, vocab_size),
        #     nn.LogSoftmax(-1)
        #     )
    def forward(self, x):
        out = self.proj(x)
        out = self.logSoftmax(out)
        return out
