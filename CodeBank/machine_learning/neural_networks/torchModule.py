import torch.nn as nn

class torchModule(nn.Module):
    """Define first, instanciate later"""
    def __new__(cls, *args, **kwargs): 
        #Remember,
        #   local.torchModule is a subclass of father torchModule
        #   you can access user implemented module by super() method
        class torchModule(cls): #child is an instance of father. Breaking Liskov Substitution principle
            def __new__(sub_cls): 
                sub_cls.__name__ = cls.__name__
                return nn.Module.__new__(sub_cls)
            def __init__(self): super().__init__(*args,**kwargs)
        return torchModule

    # def __paper__(self): raise NotImplementedError
    # def __loss__(self,*args,**kwargs): raise NotImplementedError
    # def __predict__(self,*args,**kwargs): raise NotImplementedError
    # def __doc__(self): raise NotImplementedError
    # def __forward__(self): raise NotImplementedError