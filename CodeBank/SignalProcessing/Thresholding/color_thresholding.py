
import numpy as np


class color_thresholding:
    """
    This algorithm, just thresholds all colors further away from a given color as
        |𝐼(𝑥,𝑦,𝑅,𝐺,𝐵)−𝑝𝑖𝑐𝑘𝑒𝑑_𝑐𝑜𝑙𝑜𝑟|<𝑇
    """


    def __init__(self):
        pass

    def __call__(self, img, color, T):
        R,C,_ = img.shape
        new_img = np.zeros_like(img)

        for r in range(R):
            for c in range(C):
                new_img[r,c] = np.linalg.norm(img[r, c] - color) < T
        return new_img


if __name__ == '__main__':
    test = 1
    if test == 1:
        import matplotlib.pyplot as plt
        
        pick_a_color_R_G_B  = (0, 150, 0) #@param
        T = 150 #@param {'type':'integer'}


        im_1 = color_thresholding(img, pick_a_color_R_G_B, T)
        im_1 = np.uint8(im_1)*img

        fig, ax = plt.subplots(1,2, figsize=(15,5))

        ax[0].imshow(img)
        ax[0].axis('off')
        ax[0].set_title('original');

        ax[1].imshow(im_1)
        ax[1].set_title(f'Color Threshold');
        ax[1].axis('off');  