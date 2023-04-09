import pandas as pd
from typing import Union

class fieldChooser:
    """
    DEPRECATED, use fieldFilter
    Selects how to split the columns in src and tgt"""
    def __init__(self, src:Union[list, str], tgt:Union[list,str]):
        print(fieldChooser.__doc__)
        self.srcFields = [src] if isinstance(src, str) else src
        self.tgtFields = [tgt] if isinstance(tgt, str) else tgt

    def __call__(self, data:pd.DataFrame) -> dict:
        filtered_data = {}
        filtered_data['src'] = data[self.srcFields]
        filtered_data['tgt'] = data[self.tgtFields]
        return filtered_data

    # def __call__(self, data:pd.DataFrame) -> dict:
    #     """Trying to create a multiindex column
    #     So you can access by data['src']['fields']
    #     """
    #     # data = pd.MultiIndex.from_frame(data)
    #     # pd.MultiIndex.groupby()
    #     # data.set_index(self.srcFields)
    #     # data.set_axis([['src']*self.srcFields,*self.tgtFields],axis='columns')
    #     # print()
    #     # print(data)
    #     # print()
    #     return data


if __name__ == '__main__':
    data = pd.DataFrame({'spanish':['La casa es linda','El cielo estaba despejado'],
                         'english':['the house is pretty','the sky was clear'],
                         'german': ['Das Haus ist hübsch','der Himmel war klar']})
    filter = fieldChooser(src=['spanish','english'],tgt='german')
    print(data,end='\n\n')
    filteredData = filter(data)
    print(filteredData['src'])
    print(filteredData['tgt'])
