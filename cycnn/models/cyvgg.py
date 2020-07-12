"""
Adapted and modified from
https://github.com/pytorch/vision.git
https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms

import CyConv2d_cuda
from models.cyconvlayer import CyConv2dFunction, CyConv2d


__all__ = [
    'CyVGG', 'cyvgg11', 'cyvgg11_bn', 'cyvgg13', 'cyvgg13_bn',
    'cyvgg16', 'cyvgg16_bn', 'cyvgg19_bn', 'cyvgg19',
]

class CyVGG(nn.Module):
    """
    CyVGG model 
    """
    def __init__(self, features, num_classes=10, classify=True):
        super(CyVGG, self).__init__()
        self.features = features
        self.classify = classify

        if self.classify:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        if self.classify:
            x = self.classifier(x)

        return x


def make_layers(cfg, dataset='mnist', batch_norm=False):
    layers = []
    in_channels = 1 if dataset == 'mnist' else 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = CyConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = { 
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def cyvgg11(classify=True, dataset='mnist'):
    """CyVGG 11-layer model (configuration "A")"""
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyVGG(make_layers(cfg['A'], dataset=dataset), num_classes=num_classes, classify=classify)


def cyvgg11_bn(classify=True, dataset='mnist'):
    """CyVGG 11-layer model (configuration "A") with batch normalization"""
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyVGG(make_layers(cfg['A'], dataset=dataset, batch_norm=True), num_classes=num_classes, classify=classify)


def cyvgg13(classify=True, dataset='mnist'):
    """CyVGG 13-layer model (configuration "B")"""
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyVGG(make_layers(cfg['B'], dataset=dataset), num_classes=num_classes, classify=classify)


def cyvgg13_bn(classify=True, dataset='mnist'):
    """CyVGG 13-layer model (configuration "B") with batch normalization"""
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyVGG(make_layers(cfg['B'], dataset=dataset, batch_norm=True), num_classes=num_classes, classify=classify)


def cyvgg16(classify=True, dataset='mnist'):
    """CyVGG 16-layer model (configuration "D")"""
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyVGG(make_layers(cfg['D'], dataset=dataset), num_classes=num_classes, classify=classify)


def cyvgg16_bn(classify=True, dataset='mnist'):
    """CyVGG 16-layer model (configuration "D") with batch normalization"""
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyVGG(make_layers(cfg['D'], dataset=dataset, batch_norm=True), num_classes=num_classes, classify=classify)


def cyvgg19(classify=True, dataset='mnist'):
    """CyVGG 19-layer model (configuration "E")"""
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyVGG(make_layers(cfg['E'], dataset=dataset), num_classes=num_classes, classify=classify)


def cyvgg19_bn(classify=True, dataset='mnist'):
    """CyVGG 19-layer model (configuration 'E') with batch normalization"""
    num_classes = 100 if dataset == 'cifar100' else 10
    return CyVGG(make_layers(cfg['E'], dataset=dataset, batch_norm=True), num_classes=num_classes, classify=classify)
