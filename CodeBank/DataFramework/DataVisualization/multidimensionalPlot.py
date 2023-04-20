import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

class NetworkVisualization:
    """Plots a multidimensional function into a 2d plot at a timestep t

    Renders all feedforward layers
    Sources:
        https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/#:~:text=PyTorch%20provides%20two%20types%20of%20hooks.&text=A%20forward%20hook%20is%20executed,backward%20functions%20of%20an%20Autograd.

    sources:
        https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
    """
    def __init__(self, model, obs_modules, labels):
        self.model = model
        self.labels = labels
        self.layers = OrderedDict()
        for name, layer in self.model._modules.items():
            if name in obs_modules:
                layer.register_forward_hook(self.factory_hook(name))

    def factory_hook(self, name):
        def forward_hook(module, input, output):
            self.layers[name] = output        
        return forward_hook
    def __call__(self, sample):
        output = self.model(sample)
        self.render()
    
    def render(self):
        nLayers = len(self.layers)
        fig, axs = plt.subplots(nLayers, 1, figsize=(10,10))
        for ax, (name, data) in zip(axs, self.layers.items()):
            data = data.detach()
            x_axis = torch.arange(len(data[0]))
            for sample in data:
                ax.plot(x_axis, sample,'-o')
            ax.set_title(name)
            ax.set_ylabel('activation')
        ax.set_xticks(x_axis, labels=self.labels)
        
  
if __name__ == '__main__':
    import numpy as np
    from CodeBank.machine_learning.neural_networks.deep_models.baselines import ConvFF
    from torch.utils.data import DataLoader
    from torchvision.datasets.cifar import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize

    transform = Compose([ToTensor(), Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    train_set = CIFAR10(root='./', train=True, transform=transform)

    model_1 = ConvFF()
    plot = NetworkVisualization(model_1, ['ff1', 'ff2'])
    nSamples = 3

    train_loader = DataLoader(train_set, batch_size=nSamples, shuffle=True, num_workers=2)
    sample_img, sample_tgt = next(iter(train_loader))

    plot(sample_img[0:nSamples])
    CLASS_NAMES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    plt.legend([f"True {CLASS_NAMES[sample]}" for sample in sample_tgt[:nSamples]])

    plt.figure(figsize=(10,10))
    for n in range(nSamples):
        ax = plt.subplot(1,nSamples,n+1)
        img = sample_img[n] / 2 + 0.5     # unnormalize
        img = img.numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(CLASS_NAMES[sample_tgt[n]])
        plt.axis("off")
    plt.pause(0.1)



    test = 1
    if test == 2:
        #imagine you have a NN of 2 layers and you are plotting the activations
        # of layer 1
        layer_1 = [0.5, 0.8, 0.1]
        layer_1 = np.array(layer_1)
        # of layer 2
        layer_2 = [0.6, 0.2, 0.7]#
        layer_2 = np.array(layer_2)

        plot(layer_1)
        plot(layer_2)