from cycler import cycler

from typing import Union
import matplotlib, matplotlib.pyplot as plt

from matplotlib.widgets import RangeSlider

class ContinuousPlot:
    """
    Continually update a plot.

    Features:
        - immediately updates the plot (without plt.show() call)
        - allows efficient incremental updates 
            (instead of re-plotting the entire array, it updates just the last part of it)
    TODO:
        Plot by increments and resets
    """
    def __init__(self, theme:Union[dict,str]='dark_background', xlabel='',ylabel=''):
        if   isinstance(theme, str): self.theme = plt.style.context(theme)
        elif isinstance(theme, dict): self.theme = matplotlib.rc_context(theme)
        else: self.theme = matplotlib.rc_context(self.defaultTheme())

        with self.theme:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
        plt.pause(0.001) #fist time is mandatory
        
        self.linesCollection = {}

        sliderWidth = 0.03
        pos=self.ax.get_position()
        self.sx = self.fig.add_axes((pos.x0, pos.y1-sliderWidth, pos.x1-pos.x0, sliderWidth))
        self.slider = RangeSlider(self.sx, 'x', 0, 1, valinit=(0,1))

        self.len_data = 0
        self.slider.on_changed(self.update_slider)
        self.limits = (0, 1E-5, 0, 1E-5)
    def update_slider(self, val):
        self.update_limits()
        # self.fig.canvas.draw_idle()
    
    def update_limits(self):
        x0, x1, y0, y1 = self.limits #max limits

        _x0, _x1 = self.slider.val
        x0 = _x0*self.len_data
        x1 = _x1*self.len_data+1E-5
        
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        
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
    def nextColor(self):
        self.colors = [(1.0,0.0,0.0),(0,1,0),'b']
        while True:
            for color in self.colors:
                yield color

    def addPlot(self, id):
        # color = self.nextColor()
        line, = self.ax.plot([],[])
        self.linesCollection[id] = line
    

    def __call__(self, *data, id=0):
        if id not in self.linesCollection: self.addPlot(id)
        line = self.linesCollection[id]
        self.len_data = len(data[0])

        if len(data) == 1:
            line.set_xdata(range(self.len_data))
            line.set_ydata(data[0])
            x0, x1 = 0, self.len_data
            y0, y1 = min(data[0]), max(data[0])+1E-5
        if len(data) == 2:
            line.set_data(data)
            x0, x1 = min(data[0]), max(data[0])+1E-5
            y0, y1 = min(data[1]), max(data[1])+1E-5

        self.limits = (x0, x1, y0, y1)
        self.update_limits()
        # self.ax.relim()

        # self.sx.set_visible(False)
        # self.slider.set_max(lenData)
        # self.slider.set_min(lenData-self.block_size)
        # self.slider.valstep = self.block_size/lenData
        # if self.len_data > self.block_size: 
            # self.sx.set_visible(True)
            # self.ax.set_xlim(self.len_data-self.block_size,self.len_data)
            # self.slider.set_val(((self.len_data-self.block_size)/self.len_data, 1))
            # self.slider.valstep = self.block_size/lenData

            # self.slider.set_val( (lenData-self.block_size)/lenData )
        self.ax.draw_artist(line)
        # self.ax.set_xlim(min(x), max(x) + 1E-5)
        # self.ax.set_ylim(min(y), max(y) + 1E-5)

        # if x.min()>x.max(): continue
        # if y.min()>y.max(): continue
        # self.ax.set_xlim(x.min(), x.max()+1E-5)
        # self.ax.set_ylim(y.min(), y.max()+1E-5)

        self.fig.canvas.draw() #plt with axis
        self.fig.canvas.blit(self.ax.bbox) #plt graph
        self.fig.canvas.flush_events()

    def help(self):
        print('Themes:')
        print(plt.style.available)


if __name__ == '__main__':
    plot = ContinuousPlot()
    test= 4
    if test==1:
        plot([1,2,3],    [4,5,6]);plt.pause(1)
        plot([1,2,3], [-4,-5,-6]);plt.pause(1)
    if test == 2:
        plot([4,5,3],       id=0);plt.pause(1)
        plot([-4,-5,-3],    id=1);plt.pause(1)
        plot([2,3,4],       id=0);plt.pause(1)
        plot([-2,-5,-3],    id=1);plt.pause(1)
    if test == 3:
        plot(                [5]);plt.pause(1)
        # plot(                  5);plt.pause(1)
    if test == 4:
        import numpy as np

        # data = np.random.rand(1500)
        data = np.random.rand(1000)
        # data = np.random.rand(500)
        plot(data,       id=0);plt.pause(5)