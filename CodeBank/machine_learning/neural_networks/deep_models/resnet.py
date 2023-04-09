import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    """
    Image ResNet

    Source:
        https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR
    """
    def __init__(self, block, num_blocks, output_size):
        super().__init__()
        
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, output_size)

        self.sigma  = nn.ReLU(True)
        self.pool   = nn.AvgPool2d(4)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sigma(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer4(out)
        out = self.pool(out)
        # out = .view(out)
        # out = .linear(out)
        return out
       

if __name__ == '__main__':
    from CodeBank.DataFramework.DataVisualization import ShowImg
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torch.utils.data.dataloader import DataLoader
    # import matplotlib.pyplot as plt
    # import numpy as np
    import os
    dirPath = '../../../../../../Datasets/IMG/CIFAR10/'
    dirPath = os.path.abspath(os.path.join(__file__, dirPath))+'/'

    batch_size = 32
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(dirPath, train=True, download=True, transform=transform)
    trainloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=2)
    CLASS_NAMES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    # model = ResNet()


    test = 2
    if test == 1:
        print(f"Test {test}, Visualize Data")
        images, labels = next(iter(trainloader))
        labels = [CLASS_NAMES[label] for label in labels] #Int to str
        plot = ShowImg(5,5)
        plot(images, labels)

    