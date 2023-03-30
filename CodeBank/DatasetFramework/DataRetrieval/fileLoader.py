REQUIREMENTS = """
pandas,
torch
"""

from typing import Union, Iterable
from torch.utils.data import Dataset


class FileLoader(Dataset):
    """
    Custom dataset loader from file

    Features:
        - Capacity to avoid loading in RAM, (it uses tell/seek commands)
        - Given a batch of indexes, returns a batch of samples
        - Trims the K first/last samples
    TODO:
        - Loads in Real time file
        - Reverse traversal of the file (real time purposes, as new data appends)
            - Select from last K samples (seek(end))

    REQUIREMENTS:
        - All samples are separated via \n
        
    Method for Ram shortage:
        T(Disk speed)
        M(Disk storage)
        1. Reads the entire file
            2. split the file into separated samples (sentences)
            3. Mark the begining of each sample position
        4. when __iter__(ix) calls
            5. re-open the file
            6. seek the position
            7. return sample[ix]
            
    Faster Method for Ram shortage:
        T(Ram Speed)
        M(Ram avaiability)
        1. Reads the entire file
            2. split the file into separated samples
        3. build a Pandas Dataframe
    """
    def __init__(self, path:str, use_ram=True, trim_K:int=-1):
        super().__init__()
        if trim_K <=0: trim_K = float('inf')
        self.path = path
        self.use_ram = use_ram
        self.data = []
        self.positions = []
        self.n_samples = 0
        with open(path, 'rt') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if line.isspace(): continue #also skip empty samples
                if not line: break
                if use_ram:  self.data.append( line[:-1])
                else:        self.positions.append(pos)
                self.n_samples +=1
                if self.n_samples >= trim_K: break
                
    def __len__(self):
        return self.n_samples

    def __getitem__(self, key:Union[int,Iterable[int]] ):
        if self.use_ram: return self.__getitem__ram(key)
        else:            return self.__getitem__disk(key)

    def __getitem__disk(self, key):
        with open(self.path, 'rt') as f:
            if isinstance(key, int): 
                f.seek(self.positions[key])
                out = f.readline()[:-1]
            else:                    
                out = []
                for ix in key:
                    f.seek(self.positions[ix])
                    out.append(f.readline()[:-1])
        return out
    
    def __getitem__ram(self, key):
        if isinstance(key, int): 
            return self.data[key]
        else:
            return [self.data[k] for k in key]



if __name__ == '__main__':
    trim_k = 6
    loader_ram = FileDataloader(__file__, trim_K=trim_k)
    loader_disk = FileDataloader(__file__, False, trim_K=trim_k)
    
    samples = [1,2,3,4,5]
    samples = 5
    print(loader_ram[samples], len(loader_ram))
    print(loader_disk[samples], len(loader_disk))

