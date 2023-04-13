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

class BasicBlock(nn.Module):
    """
    """       
    def __init__(self, in_planes, planes, stride=1):
        self.expansion = 1
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion)
            )
        self.sigma = nn.ReLU(True)

    def forward(self, x):
        out  = self.sigma(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out  = self.sigma(out)
        return out
    
class PreActBlock(nn.Module):
    """
    """
    def __init__(self, in_planes, planes, stride=1):
        self.expansion = 1
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride, bias=False)
                )
        self.sigma = nn.ReLU(True)
    def forward(self, x):
        out = self.sigma(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.sigma(self.bn2(out)))
        out += shortcut
        return out
    
class BottleNeck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        self.expansion = 4
        super().__init__()
        self.conv1  = nn.Conv2d(in_planes, planes,kernel_size=1, bias=False)
        self.bn1    = nn.BatchNorm2d(planes)
        self.conv2  = nn.Conv2d(planes, planes,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes)
        self.conv3  = nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1, bias=False)
        self.bn3    = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )   
        self.sigma = nn.ReLU(True)

    def forward(self, x):
        out  = self.sigma(self.bn1(self.conv1(x)))
        out  = self.sigma(self.bn2(self.conv2(out)))
        out  = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out  = self.sigma(out)
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


    test = 3
    if test == 1:
        print(f"Test {test}, Visualize Data")
        images, labels = next(iter(trainloader))
        labels = [CLASS_NAMES[label] for label in labels] #Int to str
        plot = ShowImg(5,5)
        plot(images, labels)

    if test == 2:
        print(f"Test {test}, apply model")
        preActResNet18 = ResNet(PreActBlock, [2,2,2,2]) 
        ResNet18       = ResNet(BasicBlock, [2,2,2,2]) 
        ResNet34       = ResNet(BasicBlock, [3,4,6,3]) 
        ResNet50       = ResNet(BottleNeck, [3,4,6,3]) 
        ResNet101      = ResNet(BottleNeck, [3,4,23,3]) 
        ResNet152      = ResNet(BottleNeck, [3,8,36,3]) 
