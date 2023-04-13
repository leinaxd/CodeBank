
import torch, torch.nn as nn

class ConvFF(nn.Module):
    def __init__(self, input_shape=(3,32,32), output_size=10, pDropout=0.25):
        super().__init__()
        hidden_size  = 512
        self.dropout = nn.Dropout(pDropout)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
    
        self.pool     = nn.MaxPool2d(2,2)
        self.flatten  = nn.Flatten()
        self.relu     = nn.ReLU(True)

        self.conv_out_size = self._get_conv_output(input_shape)
        self.ff1 = nn.Linear(self.conv_out_size, hidden_size)
        self.ff2 = nn.Linear(hidden_size, output_size)
        
        # self.loss_criterion = nn.CrossEntropyLoss()
        self.loss_criterion = nn.CrossEntropyLoss(reduction='none')
    def _get_conv_output(self, shape):
        """Feedfoward para saber la dimensión de la convolucional"""
        batch_size = 1
        input = torch.rand(batch_size, *shape)
        output_conv = self.forward_conv(input)
        return output_conv.size(1)

    def forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x))) #Conv1
        x = self.pool(self.relu(self.conv2(x))) #Conv2
        x = self.pool(self.relu(self.conv3(x))) #Conv3
        x = self.flatten(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.dropout(self.relu(self.ff1(x)))
        x = self.ff2(x)
        return x

    def loss(self, data, tgt):
        prediction = self(data)
        loss = self.loss_criterion(prediction, tgt)
        loss = loss.mean()
        return loss, prediction
    def predict(self, data):
        """returns (prob, prediction)"""
        return self(data).max(1)

if __name__=='__main__':
    show_params=False #@param {'type':'boolean'}
    if show_params:
        model = ConvFF()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} trainable parameters.")
        print(model)