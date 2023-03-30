from torch.utils.data import DataLoader


class BatchLoader(DataLoader):
    """
    A customized Dataloader

    Features:
        - Diferent sampling techinques
        - synchronized with seed
    """
    def __init__(self, dataset):
        pass
    def __iter__(self):
        return 0
    

# from ..sampler import mySampler
# from ..dataset import streamDataset
# from ..collateFn import myCollateFn
# from torch.utils.data import DataLoader


# class myDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size, dist):
        
#         dataset = streamDataset(dataset)
#         sampler = mySampler(dataset, batch_size, dist)
#         collateFn = myCollateFn(dataset)
#         super().__init__(dataset, batch_size, sampler=sampler, collate_fn=collateFn)

        

#     # def __iter__(self): return self.dataloader.__iter__()
#     def load_state(self, state):
#         self.sampler.load_state(state)
#     def get_state(self):
#         return self.sampler.get_state()
