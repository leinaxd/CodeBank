from typing import Callable, Iterable, Union

class DataTransformation:
    """
    Applies a function each time a sample is adquired
    """
    def __init__(self, dataset:Iterable, function:Callable):
        self.function = function
        self.dataset = dataset

    def __len__(self): return len(self.dataset)
    def __getitem__(self, key:Union[int, list]):
        if isinstance(key, int): return self.function(self.dataset[key])
        else: return [self.function(self.dataset[k]) for k in key]









