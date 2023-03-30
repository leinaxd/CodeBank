"""
This class, implements the paradigm "Define first, implement later"
    This way, every model is anytime defined but only implemented on demand.

Use Example:
    <Define first>
    class myModel(torchModel):
        model_1 = myTorchModule(..args..)
        model_2 = myTorchModule(..other.args..)
        model_3 = myOtherTorchModule(...)

    <implement later>
    models = myModel() #models is a class
    models['model_1']   #a new myTorchModule(..args..) instance
    models[2]           #a new myTorchModule(..other.args..) instance
    <or>
    model = myModel('model_3') #Just retrieve an instance of model_3
"""

from typing import Union
import torch.nn as nn
from .torchModule import torchModule
import inspect

class torchModel:
    """Asdfasdf"""
    #TODO: 
    #   1. Exigir loss, predict  y __doc__
    def __new__(cls, *args,**kwargs):
        self = super().__new__(cls) #instance
        # argNames = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]#ignore self
        # self.__kwargs__ = {name:arg for name,arg in zip(argNames, args)}
        # self.__kwargs__.update(kwargs)
        # self.__file__=cls.__init__.__code__.co_filename
        self.models={}
        self.__verbose__ = False
        for modelName, model in cls.__dict__.items():
            if not inspect.isclass(model): continue
            if issubclass(model, torchModule): self.add(modelName, model)
            # if isinstance(model, torchModule): self.add(modelName, model)

        if self.__verbose__: print(f'Avaiable models are: {self.models.keys()}')

        if args: return self[args[0]] #Just one model
        return self                   #The dict of models

    def add(self, modelName:str, model:nn.Module):
        assert issubclass(model, torchModule), f'the model {modelName} of {type(model)} should be a subclass of nn.Module'
        self.models[modelName] = model

    def __getitem__(self, nModel:Union[str,int]):
        assert self.models, 'no model is installed'
        if nModel in self.models: model = self.models[nModel]
        elif isinstance(nModel, int) and nModel > 0 and nModel <= len(self.models):
            model = list(self.models.values())[model-1] #nModel???????????????????
        else:
            raise Exception(f"'{nModel}' not recognized, try {self.models.keys()} or 1..{len(self.models)}")
        return model()#Instanciating
