
import matplotlib
import regex
from matplotlib import pyplot as plt


try: #IF COLAB
    from google.colab.patches import cv2_imshow
    from IPython.display import clear_output
    from cv2 import imread
    COLAB = True
except: 
    COLAB = False

from CodeBank.FileMagement import getPath

from .Theme.defaulTheme import defaultTheme

"""
subplot (nRows=1, nCols=2, ix=1)
    title: 'train'
subplot (nRows=1, nCols=2, ix=2)
    title: 'val'
"""


class plot:
    """
    This class holds the axes for plotting,
    
    It handles the environment for plotting (colab notebook or python script)

    __call__ 
        updates the plot
    """
    raise NotImplementedError ('not working')
    def __init__(self,
                    nRows = 1,
                    nCols = 1,
                    figName = 'fig',
                    titles:list=[],
                    dataType:object=None, 
                    path:str='', 
                    measuring:object=None, 
                    theme:object=None,
                    saveSteps=False):

        #figure config
        self.size = (nRows, nCols)
        self.figName = figName
        self.titles = titles
        assert len(titles)==self.size[0]*self.size[1], f'Please introduce a title for each plot'


        #variants
        self.theme = theme
        self.measuring = measuring
        self.dataType = dataType

        self.flags = {}
        self.flags['colab']  = COLAB
        self.flags['save']   = saveSteps
        self.flags['_save_counter'] = 0

        self.linesQueue = []
        

        self.theme = defaultTheme()

        self.path = getPath(path)
        self.fig = plt.figure()
        self.parsePlot()

    def parsePlot(self, ix):
        labels = self.titles
        color = self.theme.nextColor()
        with self.theme.defaultTheme():
            self.ax = self.fig.add_subplot(*self.size, ix)
            for _ in range(len(labels)-1):
                line, = self.ax.plot([], [], self.format, color=next(color))
                self.linesQueue.append(line)

            #x if Last row. y if First col
            if ix > (self.size[0]-1)*self.size[1]: self.ax.set_xlabel(labels[0])
            if not (ix-1) % self.size[1]         : self.ax.set_ylabel(labels[1])
            self.ax.set_title(bus.getFullName())
            self.minText = self.ax.text(0.5,1,'', c='b',bbox=self.themeExtra['bbox.facecolor'],ha='center',va='top', transform = self.ax.transAxes)
        plt.pause(0.001) #fist time is mandatory

    def __call__(self, stream, ix):
        """update call"""

        self.dataType(stream) #continuous/discrete
        
        if self.flags['colab']: self.showColab()
        if self.flags['save']: 
            self.flags['_save_counter'] = (self.flags['_save_counter']+1)%self.flags['save']
            if self.flags['_save_counter'] == 0:
                self.saveFig()

        

    def saveFig(self):
        path = self.path(self.figName+'.png')
        self.fig.savefig(path, format='png')
        return path

    def showColab(self):
        path = self.saveFig()
        clear_output()
        im = imread(path)
        cv2_imshow(im)