import regex

class measureLegend:
    """
    It consists in a legend inside the plot

    who measures the min/max/av as well any measure in the data
    """

    def __init__(self, format=''):
        self.xmin = float('inf')
        self.xmax = float('-inf')
        self.ymin = float('inf')
        self.ymax = float('-inf')
        self.format  = format
        self.parseFormat(format)
    def __call__(self):
        pass

    def parseFormat(self, string:str):
        pattern = '(min|max|av)'
        value = regex.match(pattern, string)
        self.flags['min'] = False
        self.flags['max'] = False
        self.flags['av']  = False
        for value in regex.findall(pattern, string): self.flags[value] = True
        # self.minText.set_text(f'minLoss: {self.minY:.3}')
            