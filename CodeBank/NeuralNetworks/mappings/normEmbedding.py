import torch.nn as nn
import math
class normEmbedding(nn.Module):
    """
    https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
    https://arxiv.org/abs/1608.05859
    """
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.factor = math.sqrt(embed_size)
    def forward(self, src):
        return self.embedding(src) * self.factor