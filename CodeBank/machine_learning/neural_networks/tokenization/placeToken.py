
import torch.nn as nn
from copy import deepcopy

class placeToken(nn.Module):
    """
    Place a <token> at the given position
    """
    def __init__(self, tokenIx:int, position: int):
        super().__init__()
        self.token = tokenIx
        self.position = position

    def forward(self,
                    batch:list, #TensorType['batch_size','seq_len'],
                    ):
        batch = deepcopy(batch) #don't change original batch
        for i in range(len(batch)):
            ix = self.position if 0 <= self.position else len(batch[i])+self.position+1
            batch[i].insert(ix, self.token)
        return batch

if __name__ == '__main__':
    data = [[1,2,3],[4,5],[6]]

    test = 3
    if test == 1:
        place = placeToken(float('inf'), 0)
        place(data)
        print(data)
    if test == 2:
        place = placeToken(float('inf'), -1)
        place(data)
        print(data)
    if test == 3:
        placeA = placeToken(float('inf'), 0)
        placeB = placeToken(float('inf'), -1)
        print(data)
        dataA = placeA(data)
        print(dataA)
        dataB = placeB(data)
        print(dataB)