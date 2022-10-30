
try: #IF COLAB
    from google.colab.patches import cv2_imshow
    from IPython.display import clear_output
    from cv2 import imread
    COLAB = True
except: 
    COLAB = False

from CodeBank.FileMagement import getPath

import matplotlib.figure as f

class colabPlot:
    """
    Plots a figure in google Colab Notebook
    """
    def __init__(self, fig:f.Figure, path:str=''):
        self.fig = fig
        self.path = path
        assert COLAB, f"You are not in Colab notebook i guess"
        self.path = getPath(path)
        self.figName = fig.get_label() if fig.get_label() else 'fig'
    def __call__(self):
        self.showColab()

    def saveFig(self):
        path = self.path(self.figName+'.png')
        self.fig.savefig(path, format='png')
        return path

    def showColab(self):
        path = self.saveFig()
        clear_output()
        im = imread(path)
        cv2_imshow(im)



if __name__ == '__main__':
    pass
    