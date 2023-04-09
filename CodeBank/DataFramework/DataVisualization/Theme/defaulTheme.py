from typing import Union

import matplotlib
from matplotlib import pyplot as plt
from cycler import cycler

class defaultTheme:
    def __init__(self,themeName:Union[dict,str]):
        # theme = 'seaborn-dark-palette'
        if   isinstance(themeName, str):    self.theme = plt.style.context(themeName)
        elif isinstance(themeName, dict):   self.theme = matplotlib.rc_context(themeName)
        else:                               self.theme = matplotlib.rc_context(self.defaultTheme())

    def __call__(self):
        pass

    def help(self):
        print('Themes:')
        print(plt.style.available)

    def nextColor(self):
        self.colors = [(1.0,0.0,0.0),(0,1,0),'b']
        while True:
            for color in self.colors:
                yield color
                
    def defaultTheme(self):
        # print(matplotlib.rcParams.keys())
        theme = {}
        theme['figure.facecolor'] = (0,0,0.2)
        theme['axes.titlecolor']  = (0,0.6,0.2)
        theme['axes.facecolor'] = (0,0,0.3)
        theme['axes.edgecolor'] = (0,0,0)
        theme['axes.labelcolor'] = (0,0.6,0)
        theme['axes.grid'] = True
        theme['ytick.color'] = (0.5,0.5,0)
        theme['xtick.color'] = (0.5,0.5,0)
        theme['lines.linewidth'] = 2
        theme['axes.prop_cycle']  = cycler(color = [(0.8,0,0),(0.8,0.5,0.1),(0.6,0.6,0.1)])
        self.themeExtra={}
        self.themeExtra['bbox.facecolor']  = {'facecolor':(1,1,1)}
        return theme
