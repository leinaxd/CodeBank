import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class distributionPlot:
    def __init__(self, textField:str, ax:plt.axes):
        self.textField = textField
        self.ax = ax
    def __call__(self,dataset:pd.DataFrame):

        len_count = dataset[self.textField].apply(lambda x: len(x.split()))
        sns.displot(len_count)
        

    def show(self):
        pass