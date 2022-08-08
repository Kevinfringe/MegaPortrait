import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch


class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv3D_WS(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3D_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(
                                  dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ResBlock_Custom(nn.Module):
    def __init__(self):
        super().__init__()

class ResBlock(nn.Module):
    def __init__(self, dimension, input_channels, output_channels):
        super().__init__()
        self.dimension = dimension
        self.input_channels = input_channels
        self.output_channels = output_channels
        if dimension == 2:
            self.conv_res = nn.Conv2d(self.input_channels, self.output_channels, 3, padding= 1)
            self.conv_ws = Conv2d_WS(in_channels = self.input_channels,
                                  out_channels= self.output_channels,
                                  kernel_size = 3,
                                  padding = 1)
            self.conv = nn.Conv2d(self.output_channels, self.output_channels, 3, padding = 1)
        elif dimension == 3:
            self.conv_res = nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1)
            self.conv_ws = Conv3D_WS(in_channels=self.input_channels,
                                     out_channels=self.output_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv3d(self.output_channels, self.output_channels, 3, padding=1)


    def forward(self, x):
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        return output


class Eapp1(nn.Module):
    '''
        This is the first part of the Appearance Encoder. To generate
        a 4D tensor of volumetric features vs.
    '''
    def __init__(self):
        # first conv layer, output size: 512 * 512 * 64
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=3)
        self.resblock_128 = ResBlock(dimension=2, input_channels=64, output_channels= 128)
        self.resblock_256 = ResBlock(dimension=2, input_channels=128, output_channels= 256)
        self.resblock_512 = ResBlock(dimension=2, input_channels=256, output_channels= 512)
        self.resblock3D_96 = ResBlock(dimension=3, input_channels= 1536, output_channels= 96)
        self.resblock3D_96_2 = ResBlock(dimension=3, input_channels=96, output_channels=96)
        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=1536, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)


    def forward(self, x):
        out = self.conv(x)
        out = self.resblock_128(out)
        out = self.avgpool(out)
        out = self.resblock_256(out)
        out = self.avgpool(out)
        out = self.resblock_512(out)
        out = self.avgpool(out)

        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.conv_1(out)

        # Reshape
        out = out.view(96, 16, -1, -1)

        # ResBlock 3D
        out = self.resblock3D_96(out)
        out = self.resblock3D_96_2(out)
        out = self.resblock3D_96_2(out)
        out = self.resblock3D_96_2(out)
        out = self.resblock3D_96_2(out)
        out = self.resblock3D_96_2(out)

        return out




class Eapp2(nn.Module):
'''
    This is the second part of the Appearance Encoder. To generate
    a global descriptor es that helps retain the appearance of the output
    image.
'''


class Emtn(nn.Module):




class WarpGenerator(nn.Module):




class G3d(nn.Module):




class G2d(nn.Module):

