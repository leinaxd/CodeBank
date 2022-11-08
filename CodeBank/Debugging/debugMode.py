"""
Author: Eichenbaum Daniel
Email:  leinaxd@gmail.com
Date:   6/06/2022
"""
#NOTE 1
#   Multiple experiments must be independent
#       So don't try use class atributes

class debugMode:
    """ debugMode: when debug, set a value
    Choose the right value of a variable acording to the <set> method.

    Example
        dSet_1 = debugMode('sanity', 'reduced', 'full')
        dSet_2 = debugMode('sanity', 'reduced', 'full')

        dSet_1.set('sanity')
        dSet_2.set('full')
        dataset_size = dSet_1(10, 1000, 1E12)   #return 10
        dataset_size = dSet_2(10, 1000, 1E12)   #return 1E12
    """
    def __init__(self, *args):
        self.values = args
        self.currentIx = 0
    def set(self, value):
        """Define the current value"""
        assert value in self.values, f"debugMode '{value}' not recongized, allowed values are {self.values}"
        self.currentIx = self.values.index(value)
        return self
    def __iter__(self): return self.values.__iter__()
    def __str__(self) -> str:
        txt = 'debugMode:'
        for i, value in enumerate(self.values):
            txt += f' {value}'
            txt += '[x]' if i==self.currentIx else '[ ]'
            txt += ', '
        return txt
    def __repr__(self) -> str: return self.values[self.currentIx]
    def __call__(self, *args, **kwds):
        currentKey = self.values[self.currentIx]
        if currentKey in kwds: return kwds[currentKey]
        if self.currentIx < len(args): return args[self.currentIx]
        raise Exception(f'please, fill the values for every debugging circumstances')
    def __add__(self, other):
        return self.values[self.currentIx] + other
    def __radd__(self, other):
        return other + self.values[self.currentIx]



if __name__ == '__main__':
    
    test = 2
    if test == 1:
        print(f"test {test}: string repr")
        a = debugMode('hello', 'world').set('world')
        b = 'f_'+a+'_s'
        print(b)
    if test == 2:
        print(f"test {test}: example of use")
        dSet = debugMode('UNO','DOS','TRES')
        for label in ['TRES','UNO','DOS']:
            dSet.set(label)
            print(label,dSet(1,2,3))

        
        