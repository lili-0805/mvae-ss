#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# layers.py: network layers
# author: Li Li (lili-0805@ieee.org)
#


import torch
import torch.nn as nn


class GatedConv2D(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, batch_norm=True):
        super(GatedConv2D, self).__init__()
        self.conv = nn.Conv2d(input_ch, output_ch*2, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(output_ch*2)

        nn.init.normal_(self.conv.weight, std=0.01)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        h = self.conv(x)
        if self.batch_norm:
            h = self.bn(h)
        h = h.split(h.size(1)//2, dim=1)

        return h[0] * torch.sigmoid(h[1])


class GatedDeconv2D(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, batch_norm=True):
        super(GatedDeconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_ch, output_ch*2,
                                         kernel_size, stride, padding,
                                         groups=groups, bias=bias,
                                         dilation=dilation)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(output_ch*2)

        nn.init.normal_(self.deconv.weight, std=0.01)
        if bias:
            nn.init.zeros_(self.deconv.bias)

    def forward(self, x):
        h = self.deconv(x)
        if self.batch_norm:
            h = self.bn(h)
        h = h.split(h.size(1)//2, dim=1)

        return h[0] * torch.sigmoid(h[1])


class SiLUConv2D(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, use_norm=True):
        super(SiLUConv2D, self).__init__()
        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.silu = nn.SiLU()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.GroupNorm(1, output_ch)  # Layer Normalization

        nn.init.normal_(self.conv.weight, std=0.01)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        h = self.conv(x)
        if self.use_norm:
            h = self.norm(h)
        h = self.silu(h)

        return h


class SiLUDeconv2D(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, use_norm=True):
        super(SiLUDeconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_ch, output_ch,
                                         kernel_size, stride, padding,
                                         groups=groups, bias=bias,
                                         dilation=dilation)
        self.silu = nn.SiLU()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.GroupNorm(1, output_ch)  # Layer Normalization

        nn.init.normal_(self.deconv.weight, std=0.01)
        if bias:
            nn.init.zeros_(self.deconv.bias)

    def forward(self, x):
        h = self.deconv(x)
        if self.use_norm:
            h = self.norm(h)
        h = self.silu(h)

        return h
