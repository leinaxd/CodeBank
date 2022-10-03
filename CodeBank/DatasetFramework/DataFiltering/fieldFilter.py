import pandas as pd
from typing import Union

class fieldFilter:
    """Preserve only the selected fields"""
    def __init__(self, fields:Union[list,str]):
        self.fields = [fields] if isinstance(fields, str) else fields

    def __call__(self, data:pd.DataFrame) -> dict:
        return data[self.fields]



if __name__ == '__main__':
    data = pd.DataFrame({'spanish':['La casa es linda','El cielo estaba despejado'],
                         'english':['the house is pretty','the sky was clear'],
                         'german': ['Das Haus ist hübsch','der Himmel war klar']})
    filter = fieldFilter(['spanish','german'])
    print(data,end='\n\n')
    filteredData = filter(data)
    print(filteredData)
