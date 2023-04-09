from typing import Union
import pandas as pd
import random

class maskPretraining:
    """
    TASK: Masked Language Model (MLM)
    Paper:
        BERT
    Given a sequence
        A B C D E F G
    It randomly mask with prob. p
        src: 
            A B <mask> D E F <mask>
        tgt:   
            A B C D E F G
    Application:
        Used in BERT
    """
    def __init__(self, probMask:float, field:Union[str,list], maskToken:int):
        super().__init__()
        self.probMask = probMask
        self.field = field if isinstance(field, list) else [field]
        self.maskToken = maskToken

    def __call__(self, batch:pd.DataFrame) -> pd.DataFrame:
        tgt = batch[self.field]
        src = tgt.applymap(self.sample)
        return src, tgt

    def sample(self, data:list):
        """sampling randomly"""
        lenData = len(data)
        mask = random.choices([True, False],weights=[self.probMask, 1-self.probMask], k=lenData)
        return [data[i] if mask[i] else self.maskToken for i in range(lenData)]

if __name__ == '__main__':
    maskToken = -1
    data = pd.DataFrame({'text':[[1,2,3,4,5,6,7,8,9,10],[10,20,30,40,50,60,70,80,90,100]]})
    sampler = maskPretraining(0.5,'text',maskToken)
    #2 batches x length 5
    print(data)
    test = 1
    if test == 1:
        src,tgt = sampler(data)
        print(f"{'-'*50}\nsrc:\n{src}\ntgt:\n{tgt}\n{'-'*50}\n")
