from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt, matplotlib.axes
import pandas as pd
import matplotlib

#TODO:
#   Reimplementar con matplotlib.pyplot.bar
#   pues seaborn no aporta nada...

class barPlot:
    def __init__(self, categoryField:str):

        self.categoryField = categoryField
       

    def __call__(self,
            dataset:pd.DataFrame, 
            ax:matplotlib.axes.Axes=None,
            hold_on:bool=False):
        ax = ax if ax else plt.gca()
        ax.patches = ax.patches if hold_on else [] #start fresh or overwrite
        prevPatches = len(ax.patches) #count old drawings

        category_count = dataset[self.categoryField].value_counts()
        sns.barplot(x=category_count.index, y=category_count,ax=ax)
        for i,p in enumerate(ax.patches):
            if i < prevPatches: continue #skip old patches
            ax.annotate(
                f"{category_count.index[i]}\n{p.get_height():.0f}",
                xy=(p.get_x()+p.get_width()/2, p.get_height()),
                xytext=(0,-25),
                size=13,
                color='white',
                ha='center', va='center',
                textcoords='offset points', 
                bbox={
                    'boxstyle':'round',
                    'facecolor':'none',
                    'edgecolor':'white',
                    'alpha':0.5}
                )

    def multiIndexPlot(self, 
            dataset:Dict[str,pd.DataFrame], 
            firstIndex:str=None,
            ax:matplotlib.axes.Axes=None):
        ax = ax if ax else plt.gca()

        if isinstance(dataset, dict): 
            dataset = pd.concat(dataset, axis=0) #convert into pandas.DataFrame
            dataset = dataset.reset_index()
            firstIndex = 'level_0'
        else: 
            assert firstIndex in dataset, f'Please enter firstIndex'

        # print(dataset.groupby(['level_0',self.categoryField]).groups)
        categoryOrder = dataset[self.categoryField].unique()
        dataset = dataset[[firstIndex,self.categoryField]].value_counts()
        dataset = dataset.reset_index()
        dataset = dataset.rename(columns={0:'count'})
        sns.barplot(data=dataset, x=firstIndex, y='count',hue=self.categoryField, hue_order=categoryOrder,ax=ax)
        
        for i,p in enumerate(ax.patches):
            ax.annotate(
                f"{categoryOrder[i//len(categoryOrder)]}\n{p.get_height():.0f}",
                xy=(p.get_x()+p.get_width()/2, p.get_height()),
                xytext=(0,-25),
                size=13,
                color='white',
                ha='center', va='center',
                textcoords='offset points', 
                bbox={
                    'boxstyle':'round',
                    # 'facecolor':'none',
                    'edgecolor':'white',
                    'alpha':0.5}
                )

if __name__=='__main__':
    df = sns.load_dataset("penguins") #['species','island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']
    plt.style.use('dark_background')
    test = 1

    if test == 1:
        print(f"Test {test}: multiIndex example")
        ax=plt.subplot(1,2,1)
        data = pd.DataFrame({
            'type':['train','train','train','val','val','val','test','test','test'],
            'category':['a','b','c','a','b','c','a','b','c'],
            'category_count':[2,2,1, 1,2,1, 2,2,2],
            })
        sns.barplot(data=data, x='type', y='category_count',hue='category')
        ax.set_title('expected')
        plt.legend(loc='lower left')
        ax=plt.subplot(1,2,2)
        data = {'train':pd.DataFrame({'category':['a','a','b','b','c']}),
                'val':pd.DataFrame({'category':['a','b','b','c']}),
                'test':pd.DataFrame({'category':['a','a','b','b','c','c']})
            }
        plotter = barPlot('category')
        plotter.multiIndexPlot(data, 'type')
        ax.set_title('obtained')
        plt.legend(loc='lower left')
        plt.show()
    if test == 2:
        categoryField = 'island'
        plotter = barPlot(categoryField)
        plotter.multiIndexPlot(df)

