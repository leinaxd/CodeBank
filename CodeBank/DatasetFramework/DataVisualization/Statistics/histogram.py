import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

class histogram:
    """
    ToDo
        parzen window


    <normalize>: Make the area under the histogram sum up to 1
    <histtype>: 
        'bar':  bar plot
        'step': continuous plot
    Returns: (As seen in plt.hist)
        probs: List of probabilities for each bin
        x_axis: each bin position (x-axis position)
        patches: list of figure patches
    """
    def __init__(self, 
        nBins:int, 
        range:tuple=None, 
        window='rect', 
        normalize=True,
        **plot_args):
        self.nBins = nBins
        self.range = range
        self.window = window
        self.normalize=normalize
        self.plot_args = {'fill':False}
        self.plot_args.update(plot_args)

    def __call__(self, 
        ax:plt.Axes, 
        data:np.ndarray,
        histtype='bar', #do barPlot or contour plot
        barWidth_pcn=0.95,
        bandwidth=1.
        ):
        if histtype.lower() in ['gaussian','tophat','epanechnikov','exponential','linear','cosine']: 
            return self.doKDE(ax, data, histtype, bandwidth)
        if histtype.lower() in ['bar']: 
            return self.doBar(ax, data, histtype)
        if histtype.lower() in ['step']:
            return self.doStep(ax, data, histtype)


    def doBins(self, data):
        if self.range: minData, maxData = self.range
        else:          minData, maxData = min(data), max(data)

        binWidth = (maxData-minData)/self.nBins
        bins = {center:0 for center in np.arange(minData+binWidth/2, maxData+1E-9, binWidth)}
        # bins = {center:0 for center in np.arange(minData+binWidth/2, maxData+binWidth/2, binWidth)}

        for item in data:
            for center in bins:
                if (center-binWidth/2 <= item) and (item < center+binWidth/2): #rectangular window
                    bins[center] += 1
                    break
        # print(data)
        # print(bins, minData, maxData)
        if self.normalize:
            area = sum( bins.values() ) * binWidth
            for k, v in bins.items():
                bins[k] = v/area
        return bins, binWidth

    def doBar(self, ax:plt.Axes, data:np.ndarray, histtype:str):
        if 'color' not in self.plot_args:
            color = ax._get_lines.get_next_color()
            self.plot_args.update(color = color)
        barWidth_pcn = 0.95
        bins, binWidth = self.doBins(data)
        lines = ax.bar(bins.keys(),bins.values(),width=binWidth*barWidth_pcn, color=color)
        return list(bins.values()), list(bins.keys()), lines #probs, x_axis, patches

    def doStep(self, ax:plt.Axes, data:np.ndarray, histtype:str):
        bins, binWidth = self.doBins(data)
        x, y = [None], [0]
        for pos, prob in bins.items():
            x.append(pos-binWidth/2)
            x.append(pos+binWidth/2)
            y.append(prob)
            y.append(prob)
        x[0] = x[1] #start & end with zero prob
        x.append(x[-1])
        y.append(0)
        if self.plot_args['fill']: lines = ax.fill(x, y, **self.plot_args)
        else:                      lines = ax.plot(x, y)
        # nTicks = 5
        # tick_range = max(bins.values())
        # msd = np.round_(np.log10(tick_range), 0)
        # tick_step = nTicks*10**(msd - 1) #one digit less
        
        # ticks = np.arange(0, tick_range+tick_step, tick_step)
        # ax.set_yticks(ticks)
        return list(bins.values()), list(bins.keys()), lines #probs, x_axis, patches
    
    def doKDE(self, ax:plt.Axes, data:np.ndarray, histtype:str, bandwidth:float):
        # bandwidth = 1/self.nBins
        
        x_axis = np.linspace(min(data),max(data),10*self.nBins)[:,np.newaxis]
        if isinstance(data, list): data = np.array(data)
        if len(data.shape) == 1: data = data[:,np.newaxis]
        kde     = KernelDensity(kernel=histtype,bandwidth=bandwidth).fit(data)
        logProb = kde.score_samples(x_axis)
        probs   = np.exp(logProb)
        lines = ax.fill_between(x_axis[:,0],probs,alpha=0.7)
        return probs, x_axis, lines


if __name__ == '__main__':
    test = 1
    if test == 1:
        import numpy as np
        plt.style.use('dark_background')
        size = 10000
        data_u = np.random.random(size)
        data_g = np.random.randn(size)


        print(f"test {test}")
        hist = histogram(10)
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4,2, figsize=(6.4,6.8))
        hist(ax1, data_g, histtype='bar')
        hist(ax2, data_u, histtype='bar')
        hist(ax3, data_g, histtype='step')
        hist(ax4, data_u, histtype='step')
        hist(ax5, data_g, histtype='gaussian')
        hist(ax6, data_u, histtype='gaussian')
        hist(ax7, data_g, histtype='epanechnikov')
        hist(ax8, data_u, histtype='epanechnikov')

        plt.show()

    if test == 2:
        size = 100
        data_g = np.random.randn(size,1)
        plt.style.use('dark_background')
        plt.show()