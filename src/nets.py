'''
Build 3-HMT ResNet20 model and 3-HMT CNNC model.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import layers as L


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockMULT3(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(BasicBlockMULT3, self).__init__()
        self.hmt1 = L.MultLayer3(in_channels, out_channels, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=stride, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.hmt2 = L.MultLayer3(out_channels, out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                padding = (0, 0, 0, 0, (out_channels-in_channels)//2, out_channels-in_channels-(out_channels-in_channels)//2)
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], padding, "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * out_channels)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.pool1(self.hmt1(x))))
        out = self.bn2(self.hmt2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


'''
RESNET20MULT3 structure with 3 transforms for CIFAR-10
'''
class RESNET20MULT3(nn.Module):
    def __init__(self, block=BasicBlockMULT3, num_blocks=[3,3,3], num_classes=10):
        super(RESNET20MULT3, self).__init__()

        self.in_channels = 16
        # self.hmt1 = L.MultLayer(in_channels=3, out_channels=16, bias=False)
        self.hmt1 = L.MultLayer(in_channels=3, out_channels=16, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, L.MultLayer3) or isinstance(m, L.MultLayer):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.hmt1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


'''
VGG-naga structure with 2 transforms for CIFAR-10
'''
class VGGnagaMULT(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=10):
        super(VGGnagaMULT, self).__init__()
        # block 1 -- outputs 16x16x64
        self.hmt1_1 = L.MultLayer(in_channels=3, out_channels=64)
        self.hmt1_2 = L.MultLayer(in_channels=64, out_channels=64)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 8x8x128
        self.hmt2_1 = L.MultLayer(in_channels=64, out_channels=128)
        self.hmt2_2 = L.MultLayer(in_channels=128, out_channels=128)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 4x4x256
        self.hmt3_1 = L.MultLayer(in_channels=128, out_channels=256)
        self.hmt3_2 = L.MultLayer(in_channels=256, out_channels=256)
        self.hmt3_3 = L.MultLayer(in_channels=256, out_channels=256)
        self.hmt3_4 = L.MultLayer(in_channels=256, out_channels=256)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fully connected
        self.fc6 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout()
        self.fc7 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout()
        self.fc8 = nn.Linear(1024, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, L.MultLayer):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1_1(self.hmt1_1(x)))
        x = F.relu(self.bn1_2(self.hmt1_2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2_1(self.hmt2_1(x)))
        x = F.relu(self.bn2_2(self.hmt2_2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3_1(self.hmt3_1(x)))
        x = F.relu(self.bn3_2(self.hmt3_2(x)))
        x = F.relu(self.bn3_3(self.hmt3_3(x)))
        x = F.relu(self.bn3_4(self.hmt3_4(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout1(x)       
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        x = self.fc8(x)
        return x


'''
ConvPool-CNN-C with 3 transforms for CIFAR-100
'''
class CNNCMULT3(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=100):
        super(CNNCMULT3, self).__init__()
        # block 1 -- outputs 16x16x64
        self.conv1_1 = L.MultLayer(in_channels=3, out_channels=96)
        self.conv1_2 = L.MultLayer3(in_channels=96, out_channels=96)
        self.conv1_3 = L.MultLayer3(in_channels=96, out_channels=96)
        self.bn1_1 = nn.BatchNorm2d(96)
        self.bn1_2 = nn.BatchNorm2d(96)
        self.bn1_3 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 8x8x128
        self.conv2_1 = L.MultLayer3(in_channels=96, out_channels=192)
        self.conv2_2 = L.MultLayer3(in_channels=192, out_channels=192)
        self.conv2_3 = L.MultLayer3(in_channels=192, out_channels=192)
        self.bn2_1 = nn.BatchNorm2d(192)
        self.bn2_2 = nn.BatchNorm2d(192)
        self.bn2_3 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 8x8x256
        self.conv3_1 = L.MultLayer3(in_channels=192, out_channels=192)
        self.conv3_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)
        self.conv3_3 = nn.Conv2d(in_channels=192, out_channels=num_classes, kernel_size=1)
        self.bn3_1 = nn.BatchNorm2d(192)
        self.bn3_2 = nn.BatchNorm2d(192)
        self.bn3_3 = nn.BatchNorm2d(num_classes)
        
        # average_pooling
        self.avgpool = nn.AvgPool2d((8, 8))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x