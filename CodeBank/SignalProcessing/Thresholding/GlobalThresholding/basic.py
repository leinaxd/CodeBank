

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

    nbins = 100 #@param{'type':'integer'}

    th = 150 #@param {'type':'integer'}
    ylim = 0.03 #@param {'type':'number'}
    fig, ax = plt.subplots(2,3, figsize=(15,10))

    ax[0][0].imshow(g_img, cmap='gray')
    ax[0][0].axis('off')
    ax[0][0].set_title('original');
    ax[1][0].hist(g_img.flatten(), bins=nbins, density=True);
    ax[1][0].axvline(th, color='r')
    ax[1][0].set_xlim([0, 255]);
    ax[1][0].set_ylim([0, ylim]);

    im = (g_img > th)*g_img
    im = np.uint8(im)
    f_im = im.flatten()

    ax[0][1].imshow(im, cmap='gray')
    ax[0][1].set_title(f'Background Threshold: {th}');
    ax[0][1].axis('off');
    ax[1][1].hist(f_im[f_im.nonzero()], bins=nbins, density=True);
    ax[1][1].set_xlim([0, 255]);
    ax[1][1].set_ylim([0, ylim]);

    im = (g_img <= th)*g_img
    im = np.uint8(im)
    f_im = im.flatten()

    ax[0][2].imshow(im, cmap='gray')
    ax[0][2].set_title(f'Foreground Threshold: {th}');
    ax[0][2].axis('off');
    ax[1][2].hist(f_im[f_im.nonzero()], bins=nbins, density=True)
    ax[1][2].set_xlim([0, 255]);
    ax[1][2].set_ylim([0, ylim]);