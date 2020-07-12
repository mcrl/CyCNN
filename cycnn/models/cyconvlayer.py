import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import CyConv2d_cuda

import torch.autograd as autograd

###################################################################
## cycnn module
###################################################################

class CyConv2dFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, workspace, stride=1, padding=0, dilation=1):
        ctx.input = input
        ctx.weight = weight
        ctx.workspace = workspace
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        output = CyConv2d_cuda.forward(
                input, weight, workspace, stride, padding, dilation)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = CyConv2d_cuda.backward(
                ctx.input, grad_output.contiguous(), ctx.weight, ctx.workspace,
                ctx.stride, ctx.padding, ctx.dilation)

        return grad_input, grad_weight, None, None, None, None

class CyConv2d(nn.Module):
    
    """Workspace for Cy-Winograd algorithm"""
    workspace = torch.zeros(1024 * 1024 * 1024 * 1, dtype=torch.float32).to(torch.device('cuda'))

    def __init__(self, in_channels, out_channels, kernel_size,
                             stride=1, padding=0, dilation=1):
        super(CyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        output = CyConv2dFunction.apply(input,
                                        self.weight,
                                        CyConv2d.workspace,
                                        self.stride,
                                        self.padding,
                                        self.dilation)
        return output

    def extra_repr(self):
        return 'C={}, K={}, RS={}x{} str={}, pad={}, dil={}'.format(
                self.in_channels, self.out_channels,
                self.kernel_size, self.kernel_size,
                self.stride, self.padding, self.dilation)
