import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LogRegTorch(nn.Module):
    def __init__(self):
        super(LogRegTorch, self).__init__()
        self.fc1 = nn.Linear(28*28,10)
        
    def forward(self, x):
        x = self.fc1(x)
        return torch.log_softmax(x, -1)


class DenseModel_Torch(nn.Module):
    def __init__(self, layers):
        super(DenseModel_Torch, self).__init__()
        self.layers = nn.ModuleList()  # Creates a list to hold the layers
        
        # Loop through the provided list of layers
        for i in range(len(layers) - 1):
            # Add a linear layer followed by a ReLU, except for the last layer
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        # Forward pass through the network
        for layer in self.layers:
            x = layer(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CW1 = nn.Conv2d(1, 20, 5)
        self.CW2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50,500)
        self.fc2 = nn.Linear(500, 10)
        self.maxpool = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        x = F.relu(self.CW1(x)) # relu needs to be done at some point
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = F.relu(self.CW2(x)) # relu needs to be done at some point
        x = self.maxpool(x)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet18_torch(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_torch, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion, track_running_stats=False),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
    
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
    
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
    
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNN_CIFAR10_torch(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_CIFAR10_torch, self).__init__()
        self.conv_layer_b1 = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # No change in dimension (32x32)
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # Dimension reduced to 16x16
        
        self.conv_layer_b2 = nn.Sequential(
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Dimension reduced to 8x8

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Dimension reduced to 4x4
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer_b1(x)
        x = self.conv_layer_b2(x)
        x = x.view(-1, 128 * 4 * 4)  # Flatten the features for fully connected layer
        
        x = self.fc_layer(x)
        return x

    
class VGG16_torch(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_torch, self).__init__()
        # Convolutional layers block
        self.features = nn.Sequential(
            # 1st conv block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2nd conv block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3rd conv block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4th conv block
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5th conv block
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        # Apply conv layers
        x = self.features(x)
        # Flatten
        x = torch.flatten(x, 1)
        # return x
        # Apply classifier
        x = self.classifier(x)
        return x


class VGG16_CIFAR10_torch(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR10_torch, self).__init__()

        # Convolutional layers block
        self.features = nn.Sequential(
            # 1st block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3rd block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 4th block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 5th block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # MaxPool for the 5th block, modified stride and padding
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        # Fully connected layers block
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),  # Adjusted for CIFAR-10 input size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool5(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the FC layers
        x = self.classifier(x)
        return x