import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import conv2D, Conv2d, Linear

# Code adapted from https://github.com/kuangliu/pytorch-cifar and https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride: int = 1, quantization: dict = None, winogradArgs: dict = None, miscArgs: dict = None):
        super(BasicBlock, self).__init__()

        # There's not an equivalent winograd convolution to conv2d w/ stride 2, therefore, if this happens we will first maxpool the input and then perform the convolution (winograd or std) with stride 1 
        self.ss = True if stride == 2 else False
        self.maxPool = nn.MaxPool2d(2)

        self.conv1 = conv2D(in_planes, planes, kDim = 3, quantization=quantization, winogradArgs=winogradArgs, miscArgs=miscArgs)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv2D(planes, planes, kDim = 3, quantization=quantization, winogradArgs=winogradArgs, miscArgs=miscArgs)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        # Winograd can't be applied to optimize 1x1 convolutions :(
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes, kDim=1, quantization=quantization),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        '''The main change here compare to the default `BasicBlock` is that, since stride=2 Winograds are not possible, we instead perform a maxpool. For consistency, we do this also when training a model with default convolutions.'''

        if self.ss:
            x = self.maxPool(x)

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int, multiplier: int = 1.0, quantization: dict = None, winogradArgs: dict = None, miscArgs: dict = None):
        super(ResNet, self).__init__()

        self.name = 'ResNet18Like'
        if winogradArgs['isWinograd']:
            self.name += 'F' + str(winogradArgs['F'])

        self.multiplier = multiplier
        self.in_planes = 32 # here we reduced from 64 to 32 to lower the memory requirements of the first resnet block.

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64 * self.multiplier, num_blocks[0], 1, quantization, winogradArgs, miscArgs)
        self.layer2 = self._make_layer(block, 128 * self.multiplier, num_blocks[1], 2, quantization, winogradArgs, miscArgs)
        self.layer3 = self._make_layer(block, 256 * self.multiplier, num_blocks[2], 2, quantization, winogradArgs, miscArgs)

        if winogradArgs:
            # Last residual layer has 4x4 inputs (or 6x6 if we account for 1px padding) therefore it doesn't make sense using F6 Winograds (since it requires 8x8 input tiles)...
            winogradArgs['F'] = 4 if winogradArgs['F'] == 6 else winogradArgs['F']

        self.layer4 = self._make_layer(block, 512 * self.multiplier, num_blocks[3], 2, quantization, winogradArgs, miscArgs)

        self.linear = nn.Linear(int(512 * self.multiplier)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, quantization, winogradArgs, miscArgs):
        strides = [stride] + [1]*(num_blocks-1)
        planes = int(planes)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, quantization, winogradArgs, miscArgs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(winogradArgs: dict = None, quantArgs: dict = None, miscArgs: dict = None, num_classes: int = 10, mult: int = 1.0):

    return ResNet(BasicBlock, [2,2,2,2], num_classes, winogradArgs=winogradArgs, quantization=quantArgs, miscArgs=miscArgs, multiplier=mult)