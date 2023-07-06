import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 96, 11, 4, 2)
        self.conv_layer2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv_layer3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv_layer4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv_layer5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.fc1_layer = nn.Linear(4096, 4096)
        self.fc2_layer = nn.Linear(4096, 2048)
        self.fc3_layer = nn.Linear(2048, 8)

        self.lrn = nn.LocalResponseNorm(size=5, k=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
    
    def forward(self,x):

        x = self.conv_layer1(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pooling(x)

        x = self.conv_layer2(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pooling(x)

        x = self.conv_layer3(x)
        x = self.relu(x)

        x = self.conv_layer4(x)
        x = self.relu(x)

        x = self.conv_layer5(x)
        x = self.relu(x)
        x = nn.MaxPool2d(3,3)(x)

        x = torch.flatten(x)
        x = self.fc1_layer(x)
        x = self.relu(x)
        x = self.fc2_layer(x)
        x = self.relu(x)
        x = self.fc3_layer(x)
        x = nn.Softmax()(x)
        return x

model = AlexNet()

input = torch.zeros([1,3,224,224])
output = model(input)

print(output.shape)