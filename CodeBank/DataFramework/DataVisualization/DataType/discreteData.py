
#TODO
#   Poner un mejor nombre
#   Esta funcion plotea sin borrar lo anterior. Tiene cierta memoria
class discreteData:
    """
    The discrete data is plotted as a scatter plot
    """
    def __init__(self, memory=2):
        self.reset = True

        self.memory = memory
        self.memory = 1
        self._memory = 0

        self.format = 'o'
    def __call__(self):
        pass

    def discrete_update(self, stream, clear):
        labels=list(stream.keys())
        for i, line in enumerate(self.linesQueue):#signals x, y, predict
            if clear and self.memory: 
                #Brighter for older (as they are harder to see)
                #x = (1+self._memory)/self.memory -> [0-1]
                #f(1)=1. f(0)=1. f(0.5)=ymin
                #f(x)=ax^2+bx+c. a = 4*(1-ymin), a =-b, c=1
                ymin=0.4
                x = (1+self._memory)/self.memory
                a = 4*(1-ymin)
                factor = (a*x**2-a*x+1)
                line.set_color([i*factor for i in self.colors[i]])

            x, y = stream[labels[0]], stream[labels[i+1]]
            xs,xe = x.updatedRangeIx()
            ys,ye = y.updatedRangeIx()
            s = min(xs, ys)
            e = min(xe, ye)
            line.set_xdata(x[s:e])
            line.set_ydata(y[s:e])
            if i==0 and e>s: #fit to newest
                if x.min()>x.max(): continue
                if y.min()>y.max(): continue
                self.ax.set_xlim(x.min(), x.max()+1E-5)
                self.ax.set_ylim(y.min(), y.max()+1E-5)
            self.ax.draw_artist(line)
            # ax.redraw_in_frame()
            # if clear: ax.redraw_in_frame()
            # else: ax.draw_artist(line)
        if clear:
            self._memory = (self._memory+1)%self.memory
            if not self._memory: self.fig.canvas.draw() #plt with axis
        self.fig.canvas.blit(self.ax.bbox) #plt graph
        self.fig.canvas.flush_events()
