import torch
import numpy as np

class choices:
    """
        Return the index given a matrix of probability

        normalization implemented:
            linear [l]
            softmax [s]
    """
    def __init__(self, norm='linear', dim=-1):
        self.dim = dim
        self.norm = norm

    def normalizeProbs(self, data:torch.Tensor):
        if isinstance(data,torch.Tensor): data = data.numpy()
        if self.norm.lower() in ['linear', 'l']:
            data = data/np.sum(data,-1,keepdims=True)
        if self.norm.lower() in ['softmax', 's']:
            data = np.exp(data)
            data = data/np.sum(data,-1,keepdims=True)
        return data

    def __call__(self, data:np.ndarray):
        assert len(data.shape) <= 2, f"Not implemented for len(data.shape) > 2"

        data = self.normalizeProbs(data)
        if len(data.shape) == 1: return np.random.choice(len(data), p=data)
        

        out = np.empty(data.shape[0])
        for i, row in enumerate(data):
            out[i] = np.random.choice(len(row), p=row)
        return out

if __name__ == '__main__':
    test = 4
    if test == 1:
        print(f"test {test}: single array\n")
        sampling = choices()
        data_1 = np.array([0.1,0.1,0.7,0.1])
        data_2 = np.array([1,1,0,5])
        print(f"data_1:\n\t{data_1}\ndata_2:\n\t{data_2}")
        print(f"result_1:\n\t{sampling(data_1)}\nresult_2:\n\t{sampling(data_2)}")
    if test == 2:
        print(f"test {test}: matrix array\n")
        sampling = choices()
        data = np.array([[0.1,0.1,0.7,0.1],[1,1,0,5]])
        print(f"data:\n{data}")
        for i in range(10):
            print(f"result:\n{sampling(data)}")
    if test == 3:
        print(f"test {test}: softmax\n")
        sampling = choices(norm='s')
        data = np.array([[0.1,0.1,0.7,0.1],[1,1,0,5]])
        print(f"data:\n{data}")
        for i in range(10):
            print(f"result:\n{sampling(data)}")
    if test == 4:
        import torch
        print(f"test {test}: torch compatibility\n")
        sampling = choices(norm='s')
        data = torch.tensor([[0.1,0.1,0.7,0.1],[1,1,0,5]])
        print(f"data:\n{data}")
        for i in range(10):
            print(f"result:\n{sampling(data)}")