from typing import Any
import numpy as np
import matplotlib.pyplot as plt

class OtsuThresholding:
    """
    OTSU's Algorithm
    Used to find the best global Threshold T between classes.
    Where the Image is composed of light objects on a dark background
    An alternative to the Bayesian approach (used with gaussian distributions, minimizing the avg sq error incurred in assigning pixels to two or more groups)

    Idea.
    Split the pixels into 2 classes (Background or Foreground) as well separatad as possible.
    Maximize the between-class variance. (properly thresholded classes should be distinct with respect to the intensity values of their pixelse and conversely, that a threshold giving the best separation between classes would be the best threshold)
    A good threshold should separate pixels into tight clusters
    Let  𝐿={0,1,..𝐿−1}  denote the set of L distinct integer intensity levels in a digital image of size M x N
    Let  𝑛𝑖  denote the number of pixels with intensity i. The total number, NM of pixels in the image is  𝑀𝑁=𝑛0+𝑛1+...+𝑛𝐿−1 
    The normalized histogram has components  𝑝𝑖=𝑛𝑖𝑀𝑁  where  ∑𝑝𝑖=1 

    Given a threshold  𝑇 , define two classes.
    𝐶1={(𝑥,𝑦)|𝐼(𝑥,𝑦)<=𝑇} 

    𝐶2={(𝑥,𝑦)|𝐼(𝑥,𝑦)>𝑇} 

    Compute the normalized histogram of the input image. Denote its components by  𝑝𝑖={0,1,...,𝐿−1} 

    Compute the cummulative sums,  𝑃1(𝑇)=∑𝑇𝑖=0𝑝𝑖  = P(I(x,y) <= T) for  0<𝑇<𝐿−1 

    Compute the cummulative mean,  𝑚(𝑇)=∑𝑇𝑖=0𝑖𝑝𝑖 

    Compute the global mean of the entire image,  𝑚𝐺(𝑇)=∑𝐿−1𝑖=0𝑖𝑝𝑖 

    =𝑚(𝑇=𝐿−1) 

    Compute the between-class variance term,
    𝜎2𝐵(𝑇)=𝑃1(𝑚1−𝑚𝐺)2+𝑃2(𝑚2−𝑚𝐺)2=𝑃1𝑃2(𝑚1−𝑚2)2=𝑚𝐺𝑃1−𝑚𝑃1(1−𝑃1) 

    Note 1:
    𝑃1𝑚1+𝑃2𝑚2=𝑚𝐺 
    𝑃1+𝑃2=1 

    𝑚1(𝑇)=∑𝑇𝑖=0𝑖𝑃(𝑖|𝑖∈𝑐1)   =∑𝑇𝑖=0𝑖𝑃(𝑖∈𝑐1|𝑖)=1𝑃(𝑖)𝑃(𝑐1)=1𝑃1(𝑇)∑𝑇𝑖=0𝑖𝑝𝑖 
    𝑚2(𝑇)=1𝑃2(𝑇)∑𝐿−1𝑖=𝑇+1𝑖𝑝𝑖 

    Note 2:
    The farther the means  𝑚1  and  𝑚2  are, the greater the  𝜎2𝐵  variance will be. Implying that the between class variance is a measure of separability between classes.

    Obtain the Otsu's threshold  𝑇∗ . If the maximum is not unique, obtain  𝑇∗  by averaging values of  𝑇  corresponding to the various maxima detected.
    𝑇∗=𝑎𝑟𝑔𝑚𝑎𝑥𝑇∈[0,𝐿−1] 𝜎2𝐵(𝑇) 

    So Otsu's criterion is to maximize the between-class variance
    Compute the global variance  𝜎2𝐺=∑𝐿−1𝑖=0(𝑖−𝑚𝐺)2𝑝𝑖 
    Compute the separability measure  𝜂=𝜎2𝐵/𝜎2𝐺 
    The measure  𝜎2𝑏/𝜎2𝑎  is a good measure of separability
    """
    def __init__(self):
        pass
    def __call__(self):
        pass

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import urllib.request
    plt.style.use('dark_background')


    url = "https://github.com/leinaxd/NOTES/raw/main/digital_image_processing/notebooks/background.jpg"
    with urllib.request.urlopen(url) as url:
        img = plt.imread(url, format='jpg')
    plt.imshow(img)
    plt.axis('off')

    print(img.shape)

    r_mask = img[:, :, 0]
    g_mask = img[:, :, 1]
    b_mask = img[:, :, 2]

    #turn image to gray
    g_img = r_mask*0.2989 + g_mask*0.5870 + b_mask*0.1140
    u_img = np.uint8(g_img)
    nbins = 50 #@param {'type':'integer'}

    #Algorithm
    # step 1
    h_im, p_im = np.histogram(g_img.flatten(), bins=nbins,density=True, range=[0,255])
    w_bar = p_im[1:]-p_im[:-1] #width of each histogram bar
    intensities = np.zeros_like(p_im) #x axis
    intensities[1:] = np.cumsum(w_bar)
    p_i = np.zeros_like(p_im) #adds zero prob to intensity < 0
    p_i[1:] = h_im * w_bar #prob = area of each histogram bar

    # print(len(h_im))
    # step 2
    P_1 = np.cumsum(p_i)
    # plt.hist(g_img.flatten(), density=True, bins=nbins, alpha=0.2)

    #step 3
    m = np.cumsum(intensities*p_i)

    #step 4
    m_G = m[-1]

    #step 5
    sigma_B = (m_G * P_1-m)**2/(1E-10+P_1*(1-P_1))

    #step 6
    ix_star = np.argmax(sigma_B)
    T_star  = intensities[ix_star]

    #step 7
    sigma_G = np.sum((intensities-m_G)**2 * p_i)

    #step 8
    eta = sigma_B[ix_star]/sigma_G

    #Others
    m_1 = m[ix_star]/P_1[ix_star]
    m_2 = (m_G - m[ix_star])/(1-P_1[ix_star])

    im_1 = (g_img > T_star)*g_img
    im_2 = (g_img <=T_star)*g_img
    im_1 = np.uint8(im_1)
    im_2 = np.uint8(im_2)

    fig, ax = plt.subplots(2,3, figsize=(15,10))
    ax[0][0].bar(p_im[:-1], h_im, align='edge', width=1*w_bar[0])
    ax[0][0].set_title('$p_i$: PDF')
    ax[0][0].text(T_star-15, 0.006, 'T*',color='r')
    ax[0][0].axvline(T_star, color='r')
    ax[0][0].text(m_G+5, 0.008, 'm_G',color='g')
    ax[0][0].axvline(m_G, color='g', linestyle='--')
    ax[0][0].text(m_1+5, 0.008, 'm_1',color='g')
    ax[0][0].axvline(m_1, color='g', linestyle='--')
    ax[0][0].text(m_2+5, 0.008, 'm_2', color='g')
    ax[0][0].axvline(m_2, color='g', linestyle='--')


    ax[0][1].plot(intensities, P_1)
    ax[0][1].set_title('$P_1$: cummulative distribution')

    ax[0][2].plot(intensities, sigma_B/sigma_G)
    ax[0][2].set_title('$\eta = \sigma^2_B/\sigma^2_G$')
    ax[0][2].text(T_star-15, 0.006, 'T*',color='r')
    ax[0][2].axvline(T_star, color='r')

    ax[1][0].imshow(g_img, cmap='gray')
    ax[1][0].axis('off')
    ax[1][0].set_title('original');

    ax[1][1].imshow(im_1, cmap='gray')
    ax[1][1].set_title(f'Background');
    ax[1][1].axis('off');

    ax[1][2].imshow(im_2, cmap='gray')
    ax[1][2].set_title(f'Foreground');
    ax[1][2].axis('off');