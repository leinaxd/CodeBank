from typing import Iterable

class MergeLoaders:
    """
    Merges multiple sources of dataset loaders
    """

    def __init__(self, *sources:Iterable[Iterable]):
        lengths = [len(src) for src in sources]
        assert len(set(lengths)) <= 1, 'all dataset must have same length'
        self.sources = sources
    def __len__(self): return len(self.sources[0])
    def __getitem__(self, key:int):
        return [src[key] for src in self.sources]
    


if __name__ =='__main__':
    a = {1:'one',2:'two',3:'three'}
    b = {1:'uno',2:'dos',3:'tres'}
    
    dataset = MergeLoaders(a,b)

    print(dataset[2])