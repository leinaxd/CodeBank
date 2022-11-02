"""
Author: Eichenbaum, Daniel 1/11/2022
"""

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
    ver = '1.0.1'
    def __init__(self, path:str=''):
        self.path = path
        assert COLAB, f"You are not in Colab notebook i guess"
        self.path = getPath(path)

    def __call__(self, fig:f.Figure, clear:bool=False):
        """
        Saves the fig in path/<fig_name>.png, then show it in colab notebook
        """
        path = self.saveFig(fig)
        if clear: clear_output()
        im = imread(path)
        cv2_imshow(im)

    def saveFig(self, fig:f.Figure):
        figName = fig.get_label() if fig.get_label() else 'fig'
        path = self.path(figName+'.png')
        fig.savefig(path, format='png')
        return path




if __name__ == '__main__':
    pass
    