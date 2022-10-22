
class continuousData:
    """
    The continuous data is plottes as a line plot
    """
    def __init__(self):
        self.format = '-'

    def __call__(self):
        pass


    def continuous_update(self, stream):
        labels=list(stream.keys())
        for i, line in enumerate(self.linesQueue):#signals x, y, predict
            x, y = stream[labels[0]], stream[labels[i+1]]
            line.set_xdata(x)
            line.set_ydata(y)
            self.ax.draw_artist(line)       
            if x.min()>x.max(): continue
            if y.min()>y.max(): continue
            self.ax.set_xlim(x.min(), x.max()+1E-5)
            self.ax.set_ylim(y.min(), y.max()+1E-5)
        self.fig.canvas.draw() #plt with axis
        self.fig.canvas.blit(self.ax.bbox) #plt graph
        self.fig.canvas.flush_events()