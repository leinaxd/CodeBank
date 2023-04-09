from typing import List

class slidingWindow:
    """
    slidingWindow

    Given an original list of items, 
        returns a batch of list,
        containing a list of 

    <window_size> number of items truncated from
    <overlap>: Porcentage of overlapped words   
    """
    def __init__(self, window_size:int, overlap:float, power:bool=True):
        self.window_size = window_size
        self.offset = int(overlap*window_size)
        self.min_len = self.window_size*overlap #at least do half window
        self.power = power

    def __call__(self, sample:list) -> List[list]:
        if not self.power: return sample
        result = []
        s=0
        if len(sample) <= self.min_len: result.append(sample) #for texts smaller than the window
        while s < len(sample)-self.min_len:
            e = s + self.window_size
            result.append(sample[s:e])
            s = e-self.offset
        return result


if __name__ == '__main__':
    A = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    window_size = 4
    overlap  = 0.1
    window = slidingWindow(window_size,overlap,True)
    
    print(window(A))

