from typing import Callable
import numpy as np


class blockProc:
    """
    Let be an image of size NxM, 
    Applies a function <func> to each (sx,sy) block of the image <img> 
    returns the concatenated version of that blocked transfomed image.
    """

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, img:np.ndarray, func:Callable):
        assert callable(func), 'func must be a callable'
        R = img.shape[0]//self.block_size[0] #num of row blocks
        C = img.shape[1]//self.block_size[1] #num of col blocks
        new_img = np.zeros_like(img)
        re = 0
        for r in range(R):
            rs = re
            re   = rs + self.block_size[0]
            ce = 0
            for c in range(C):
                cs = ce
                ce = cs + self.block_size[1]
                # print(self.block_size, img.shape)
                # print(rs,re,cs,ce)
                new_img[rs:re, cs:ce] = func(img[rs:re,cs:ce])

        return new_img



if __name__ == '__main__':
    test = 1
    if test == 1:
        import matplotlib.pyplot as plt
        img = np.array([[ 1,2,3,4,5,6,7,8,9,10],
                        [10,11,12,13,14,15,16,17,18,19],
                        [20,21,22,23,24,25,26,27,28,29],
                        [30,31,32,33,34,35,36,37,38,39],
                        [40,41,42,43,44,45,46,47,48,49],
                        [50,51,52,53,54,55,56,57,58,59],
                        [60,61,62,63,64,65,66,67,68,69],
                        [70,71,72,73,74,75,76,77,78,79],
                        [80,81,82,83,84,85,86,87,88,89],
                        [90,91,92,93,94,95,96,97,98,99]])

        def func(img:np.ndarray):
            return np.array([[img[0][0], img[0][1]],[-1*img[1][0],10*img[1][1]]])
        
        
        Proc = blockProc([2,2], func)


        cropped = Proc(img)

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(cropped)
        # plt.show()
        print(img)
        print(cropped)