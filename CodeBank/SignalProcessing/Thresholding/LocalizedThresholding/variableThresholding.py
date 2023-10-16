
import numpy as np

class variable_thresholding:
    """
    Variable Thresholding

    factors such as noise or non-uniform ilumination play a major role in the performance of a thresholding algorithm. It can be shown that smoothing or using the edge information can help grouping classes.
    Let's introduce the variable thresholding approach, which in its simplest form you compute a threshold for each pixel  (𝑥,𝑦)  over a neighborhood of that point.
    In its simplest form, we can use the mean  𝑚𝑥𝑦  and the standard deviation  𝜎𝑥𝑦  in the neighborhood  𝑆𝑥𝑦  as descriptors of the average intensity and contrast.

    The following are common forms of Thresholding  𝑇𝑥𝑦  based on local image properties.
        𝑇𝑥𝑦=𝑎𝜎𝑥𝑦+𝑏𝑚𝑥𝑦 
        𝑇𝑥𝑦=𝑎𝜎𝑥𝑦+𝑏𝑚𝐺 

    The segmented image is computed as:
        𝑔(𝑥,𝑦)={1 𝑖𝑓 𝐼(𝑥,𝑦)≤𝑇𝑥𝑦} 

    A significant improvement that can be done is to generalize towards the predicate  𝑄  so
        𝑔(𝑥,𝑦)={1 𝑖𝑓 𝑄(𝑙𝑜𝑐𝑎𝑙 𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠)𝑖𝑠𝑇𝑟𝑢𝑒} 

    For example
        𝑄(𝑚𝑥𝑦,𝜎𝑥𝑦)={𝑇𝑟𝑢𝑒 𝑖𝑓 𝐼(𝑥,𝑦)>𝑎𝜎𝑥𝑦 𝐴𝑁𝐷 𝐼(𝑥𝑦)>𝑏𝑚𝑥𝑦 

    More Examples:
        𝑔1(𝑥,𝑦)={1 𝑖𝑓 𝐼(𝑥,𝑦)>𝜇𝑥𝑦+2𝜎𝑥𝑦 
        𝑔2(𝑥,𝑦)={1 𝑖𝑓 𝐼(𝑥,𝑦)>𝜇𝑥𝑦 
        𝑔3(𝑥,𝑦)={1 𝑖𝑓 |𝐼(𝑥,𝑦)−𝜇𝑥𝑦|>2𝜎𝑥𝑦 
        𝑔4(𝑥,𝑦)={1 𝑖𝑓 𝐼(𝑥,𝑦)>𝜇𝑥𝑦+2𝜎𝑥𝑦 𝐴𝑁𝐷 𝐼(𝑥,𝑦)>𝑇
    """
    def __init__(self):
        pass
    def __call__(self, img, Sxy, local_func, **args):
        Sx,Sy = Sxy #x,y shape of the neighborhood
        R,C = img.shape
        back_img = np.zeros_like(img)
        fore_img = np.zeros_like(img)

        for r in range(R):
            for c in range(C):
                rs = r-Sx//2 if r-Sx//2>0 else 0
                re = r+Sx//2+1
                cs = c-Sy//2 if c-Sy//2>0 else 0
                ce = c+Sy//2+1
                back_img[r,c] = local_func(img[rs:re, cs:ce], (r-rs,c-cs), **args)
                fore_img[r,c] = 1-back_img[r,c]
        return back_img, fore_img

    def local_threshold(block, Ixy, a=1, b=2, T=125):
        mu_xy = np.mean(block)
        std_xy = np.std(block)
        return 1 if a*std_xy +b*mu_xy > T else 0


if __name__ == '__main__':  
    test = 1

    if test == 1:
        a = np.array([  [1,2,3],
                        [4,5,6],
                        [7,8,9]])

        R,C = a.shape
        Sx, Sy = (2, 2)
        for r in range(R):
            for c in range(C):
                #For each pixel.
                rs = r-Sx//2 if r-Sx//2>0 else 0
                re = r+Sx//2+1
                cs = c-Sy//2 if c-Sy//2>0 else 0
                ce = c+Sy//2+1

                print(r,c, end=', ')
                print(a[r,c], end=', ') #I(x,y)
                print((r-rs, c-cs), end=': ') #Point relative to new neighborhood
                print(a[rs:re, cs:ce] )     
    if test == 2:
        import matplotlib.pyplot as plt
        
        thresholder = variable_thresholding()

        neighborhood_shape = (10,10) #@param
        T = 50 #@param {'type':'integer'}
        a = 1 #@param {'type':'number'}
        im_1, im_2 = thresholder(g_img, neighborhood_shape, thresholder.local_threshold)
        im_1 = np.uint8(im_1)*255
        im_2 = np.uint8(im_2)*255

        fig, ax = plt.subplots(1,3, figsize=(15,5))

        ax[0].imshow(g_img, cmap='gray')
        ax[0].axis('off')
        ax[0].set_title('original');

        ax[1].imshow(im_1, cmap='gray')
        ax[1].set_title(f'Background Localized Threshold');
        ax[1].axis('off');

        ax[2].imshow(im_2, cmap='gray')
        ax[2].set_title(f'Foreground Localized Threshold');
        ax[2].axis('off');
        b = 2 #@param {'type':'number'}