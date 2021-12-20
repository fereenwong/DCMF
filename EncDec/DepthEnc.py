import torch
from torch import nn
from parameter import *


class DepthEncoder(nn.Module):
    def __init__(self, strides, out_channels, dep_channel=32):
        super(DepthEncoder, self).__init__()
        self.dep_channel = dep_channel

        self.StemNet = nn.Sequential(
            nn.Conv2d(1, self.dep_channel * 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.dep_channel * 2, self.dep_channel * 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.dep_channel * 2, self.dep_channel * 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.dep_channel * 2, self.dep_channel * 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.dep_channel * 2, self.dep_channel, 3, 1, 1))

        self.stages = nn.Sequential(*[self.make_layer(s, out_channel)
                                      for s, out_channel in zip(strides, out_channels)])

    def make_layer(self, stride, out_channel):
        kernel = min(stride * 2 + 1, 9)
        dilation = 2 if stride > 1 else 1
        padding = (1 - stride + dilation * (kernel - 1))
        padding = padding // 2 + padding % 2

        conv1 = nn.Conv2d(self.dep_channel, self.dep_channel, 1)
        relu = nn.LeakyReLU(0.1, True)
        conv2 = nn.Conv2d(self.dep_channel, out_channel, kernel, stride, padding, dilation)
        return nn.Sequential(*[conv1, relu, conv2])

    def forward(self, x):
        x = self.StemNet(x)
        feats = [self.stages[i](x) for i in range(len(self.stages))]
        return feats