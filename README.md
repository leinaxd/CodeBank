# CodeBank
A library for code reutilization

# Software Design

In order to decouple/isolate i suggest to use the following convention:

Recall the design principle called "Dependency Injection".
```
class yourFunction:
    """
    yourFunction DOC
    """
    def __init__(self, *args,**kwargs):
        """
        DEFINE THE STATIC BEHAVIOUR:
            Here you initialize your fixed parameters, or select a fixed functionality
        """
    def __call__(self, *args, **kwargs):
        """
        DEFINE THE DYNAMIC BEHAVIOUR:
            Here you choose editable parameters, or select some functionality on-demand
        """
    def method(self):
        """
        EXECUTION:
            Here you query the model with custom funcionality
        """
```



## Owner:
- Author: Eichenbaum, Daniel. eichenbaum.daniel@gmail.com
- Collaborators: None

## Description:
- Version: 1.0.1 
- Starting Date: 13/08/2022
- Releasing Date: \<ukn\>


