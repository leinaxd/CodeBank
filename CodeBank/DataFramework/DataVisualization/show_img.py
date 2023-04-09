from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np

class ShowImg:
    """
    Shows a batch of images
    """

    def __init__(self, nRows, nCols):
        # self.fig = plt.figure(figsize=(10, 10))
        with plt.style.context('dark_background'):
            self.fig = plt.figure()
        self.nRows = nRows
        self.nCols = nCols
    def __call__(self, images:Iterable, titles:Iterable):
        with plt.style.context('dark_background'):
            for n in range(self.nRows * self.nCols):
                ax = plt.subplot(self.nRows, self.nCols, n+1)
                img = images[n] / 2 + 0.5     # unnormalize
                img = img.numpy()
                plt.imshow(np.transpose(img, (1, 2, 0)))
                plt.title(titles[n])
                plt.axis("off")
        plt.pause(2)
        
if __name__=='__main__':
    import numpy as np
    import os
    dirPath = '../../../../../Datasets/IMG/CIFAR10/'
    dirPath = os.path.abspath(os.path.join(__file__, dirPath))+'/'

    print(f"\n{dirPath}")




    test = 1
    if test == 1:
        print(f"Test {test}, Visualize CIFAR10")
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import Compose, ToTensor, Normalize
        from torch.utils.data.dataloader import DataLoader

        batch_size = 32
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = CIFAR10(dirPath, train=True, download=True, transform=transform)
        trainloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=2)
        CLASS_NAMES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        plot = ShowImg(5,5)
        image_batch, labels = next(iter(trainloader))
        labels = [CLASS_NAMES[label] for label in labels]
        plot(image_batch, labels)
        plt.show()