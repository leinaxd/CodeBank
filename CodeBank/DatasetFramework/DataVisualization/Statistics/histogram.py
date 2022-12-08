import matplotlib.pyplot as plt
import numpy as np

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
        self.plot_args = plot_args

    def __call__(self, 
        ax:plt.Axes, 
        data:np.ndarray,
        histtype='bar', #do barPlot or contour plot
        barWidth_pcn=0.95,
        ):
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
        if 'color' not in self.plot_args:
            color = ax._get_lines.get_next_color()
            self.plot_args.update(color = color)

        return list(bins.values()), list(bins.keys()), self.doPlot(ax, bins, binWidth, histtype, barWidth_pcn) #probs, x_axis, patches

    def doPlot(self, ax:plt.Axes, bins:dict, binWidth, histtype, barWidth_pcn):
        if histtype == 'bar':
            lines = ax.bar(bins.keys(),bins.values(),width=binWidth*barWidth_pcn, **self.plot_args,)
        elif histtype == 'step':
            x, y = [None], [0]
            for pos, prob in bins.items():
                x.append(pos-binWidth/2)
                x.append(pos+binWidth/2)
                y.append(prob)
                y.append(prob)
            x[0] = x[1] #start & end with zero prob
            x.append(x[-1])
            y.append(0)
            lines = ax.plot(x, y, **self.plot_args)
        
        # nTicks = 5
        # tick_range = max(bins.values())
        # msd = np.round_(np.log10(tick_range), 0)
        # tick_step = nTicks*10**(msd - 1) #one digit less
        
        # ticks = np.arange(0, tick_range+tick_step, tick_step)
        # ax.set_yticks(ticks)
        return lines
    



if __name__ == '__main__':
    test = 1
    if test == 1:
        import numpy as np
        plt.style.use('dark_background')
        size = 10000
        uniform = np.random.random(size)*100
        gaussian = np.random.randn(size)

        data_g = gaussian
        data_u = uniform

        print(f"test {test}")
        hist = histogram(10)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        hist(ax1, data_g, fill=True)
        hist(ax2, data_u, fill=True)
        hist(ax3, data_g, fill=False)
        hist(ax4, data_u, fill=False)

        plt.show()
