import torch
import numpy as np

class choices:
    """
    DEPRECATED:
       SEE TORCH.MULTINOMIAL

       
        Return the index given a matrix of probability

        normalization implemented:
            linear [l]
            softmax [s]
    """
    def __init__(self, norm='linear', dim=-1):
        assert dim ==-1, f"dim is not implemented"
        self.dim = dim
        self.norm = norm

        self.sumFunc = None
        self.expFunc = None
        self.choiceFunc = None
        self.emptyFunc = None

    def detectFramework(self, data):
        if isinstance(data, np.ndarray):
            self.sumFunc = np.sum
            self.expFunc = np.exp
            self.choiceFunc = lambda x: np.random.choice(len(data), p=data)
            self.emptyFunc = np.empty
        if isinstance(data, torch.Tensor):
            self.sumFunc = torch.sum
            self.expFunc = torch.exp
            self.choiceFunc = lambda x: torch.multinomial(x, num_samples=1)
            self.emptyFunc = torch.empty
    
    def normalizeProbs(self, data:torch.Tensor):
        if self.norm.lower() in ['linear', 'l']:
            data = data/self.sumFunc(data,-1,keepdims=True)
        if self.norm.lower() in ['softmax', 's']:
            data = self.expFunc(data)
            data = data/self.sumFunc(data,-1,keepdims=True)
        return data


    def __call__(self, data):
        assert len(data.shape) <= 2, f"Not implemented for len(data.shape) > 2"
        self.detectFramework(data)
        data = self.normalizeProbs(data)

        if len(data.shape) == 1: return self.choiceFunc(data)
        

        out = self.emptyFunc(data.shape[0])
        for i, row in enumerate(data):
            out[i] = self.choiceFunc(row)

        return out

if __name__ == '__main__':
    test = 2
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
    if test == 5:
        data = torch.tensor([[0.1,0.1,0.7,0.1],[1,1,0,5]])
        print(data)
        print(data.multinomial(20,replacement=True))
