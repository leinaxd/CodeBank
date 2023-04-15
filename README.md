# CodeBank
A library for code reutilization

# Software Design

In order to decouple/isolate i suggest to use the following convention:

Recall the design principle called "Dependency Injection".
```
# your_model.py
class YourModel:
    """
    YourModel DOC (i.e. latex)
        y_n = x_n + x_{n-1} + x_{n-2} + ...
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
            You are also allowed to query the model with custom funcionality
        """

if __name__=='__test__':
    test = 1
    if test == 1:
        import OtherModel
        print(f"test {test} Applying some model")
        dependency = OtherModel()
        model = YourModel(dependency)
        out   = model([1,2,3]) #apply data
        print(out)

```



## Owner:
- Author: Eichenbaum, Daniel. eichenbaum.daniel@gmail.com
- Collaborators: None

## Description:
- Version: 1.0.1 
- Starting Date: 13/08/2022
- Releasing Date: \<ukn\>


