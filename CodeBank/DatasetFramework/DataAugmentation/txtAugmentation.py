import pandas as pd

from CodeBank.DatasetFramework.DataAugmentation.transformations import txtRandomSampling
 

class txtAugmentation:
    """
    
    transformations: Dict[type:str, prob:float]
        -synonyms

    <HuggingFaceLM>: Callable [string with [mask]-> string]
    """
    def __init__(self, transformations:dict, HuggingFaceLM:str, maskToken:str):

        self.sampling = txtRandomSampling(transformations['synonyms'], maskToken)
        self.modelLM = HuggingFaceLM
        
    def __call__(self, data: pd.Series):
        return self.sampling(data)
        # return data


if __name__ =='__main__':
    test = 1
    data = pd.DataFrame({
            'corpus':['This is an example text', 'Esto es otro texto de ejemplo']
        })
    if test == 1:
        print(f"test {test}: ")
        transformations={'synonyms':0.5}
        augmentation = txtAugmentation(transformations, None,'[MASK]')

        newData = augmentation(data['corpus'])
        print(newData)
