"""
Adapted and modified from
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import CyConv2d_cuda
from models.cyconvlayer import CyConv2dFunction, CyConv2d

__all__ = ['CyResNet', 'cyresnet20', 'cyresnet32', 'cyresnet44', 'cyresnet56', 'cyresnet110', 'cyresnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        """CyConv2d instead of nn.Conv2d"""
        self.conv1 = CyConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = CyConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    #  nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     CyConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CyResNet(nn.Module):
    def __init__(self, block, num_blocks, dataset='mnist', num_classes=10):
        super(CyResNet, self).__init__()
        self.in_planes = 16

        in_channels = 1 if dataset == 'mnist' else 3

        """CyConv2d instead of nn.Conv2d"""
        self.conv1 = CyConv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def cyresnet20(dataset='mnist'):
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyResNet(BasicBlock, [3, 3, 3], dataset=dataset, num_classes=num_classes)


def cyresnet32(dataset='mnist'):
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyResNet(BasicBlock, [5, 5, 5], dataset=dataset, num_classes=num_classes)


def cyresnet44(dataset='mnist'):
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyResNet(BasicBlock, [7, 7, 7], dataset=dataset, num_classes=num_classes)


def cyresnet56(dataset='mnist'):
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyResNet(BasicBlock, [9, 9, 9], dataset=dataset, num_classes=num_classes)


def cyresnet110(dataset='mnist'):
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyResNet(BasicBlock, [18, 18, 18], dataset=dataset, num_classes=num_classes)


def cyresnet1202(dataset='mnist'):
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyResNet(BasicBlock, [200, 200, 200], dataset=dataset, num_classes=num_classes)
