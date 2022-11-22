#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# modules.py: network modules
# author: Li Li (lili-0805@ieee.org)
#


import torch
import torch.nn as nn
from mvae_ss.nn.layers import GatedConv2D, GatedDeconv2D
from mvae_ss.nn.layers import SiLUConv2D, SiLUDeconv2D


class CVAE_Encoder(nn.Module):
    def __init__(self, n_freq, n_label):
        super(CVAE_Encoder, self).__init__()
        self.conv1 = GatedConv2D(n_freq+n_label, n_freq//2,
                                 (1, 5), (1, 1), (0, 2))
        self.conv2 = GatedConv2D(n_freq//2+n_label, n_freq//4,
                                 (1, 4), (1, 2), (0, 1))
        self.conv3 = nn.Conv2d(n_freq//4+n_label, n_freq//8*2,
                               (1, 4), (1, 2), (0, 1))

        nn.init.normal_(self.conv3.weight, std=0.01)
        nn.init.zeros_(self.conv3.bias)

    def concat_xy(self, x, y):
        n_h, n_w = x.shape[2:4]
        h = torch.cat((
            x, y.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_h, n_w)), dim=1)

        return h

    def forward(self, x, c):
        x = x.permute(0, 2, 1, 3)

        h = self.conv1(self.concat_xy(x, c))
        h = self.conv2(self.concat_xy(h, c))
        h = self.conv3(self.concat_xy(h, c))
        mu, logvar = h.split(h.size(1)//2, dim=1)

        return mu, logvar


class CVAE_Decoder(nn.Module):
    def __init__(self, n_freq, n_label):
        super(CVAE_Decoder, self).__init__()
        self.deconv1 = GatedDeconv2D(n_freq//8+n_label, n_freq//4,
                                     (1, 4), (1, 2), (0, 1))
        self.deconv2 = GatedDeconv2D(n_freq//4+n_label, n_freq//2,
                                     (1, 4), (1, 2), (0, 1))
        self.deconv3 = nn.ConvTranspose2d(n_freq//2+n_label, n_freq,
                                          (1, 5), (1, 1), (0, 2))

        nn.init.normal_(self.deconv3.weight, std=0.01)
        nn.init.zeros_(self.deconv3.bias)

    def concat_xy(self, x, y):
        n_h, n_w = x.shape[2:4]
        h = torch.cat((
            x, y.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_h, n_w)), dim=1)

        return h

    def forward(self, z, c):
        h = self.deconv1(self.concat_xy(z, c))
        h = self.deconv2(self.concat_xy(h, c))
        h = self.deconv3(self.concat_xy(h, c))

        h = h.permute(0, 2, 1, 3)

        return torch.clamp(h, min=-80.)


class ACVAE_Classifier(nn.Module):
    def __init__(self, n_freq, n_label):
        super(ACVAE_Classifier, self).__init__()
        self.conv1 = GatedConv2D(n_freq, n_freq//2,
                                 (1, 5), (1, 1), (0, 2))
        self.conv2 = GatedConv2D(n_freq//2, n_freq//4,
                                 (1, 4), (1, 2), (0, 1))
        self.conv3 = GatedConv2D(n_freq//4, n_freq//16,
                                 (1, 4), (1, 2), (0, 1))
        self.conv4 = nn.Conv2d(n_freq//16, n_label, 
                               (1, 4), (1, 2), (0, 1))
        self.softmax = nn.Softmax(dim=1)

        nn.init.normal_(self.conv4.weight, std=0.01)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)

        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)

        h = self.softmax(h)
        h = torch.prod(h, dim=3)
        prob = torch.squeeze(torch.clamp(h, 1.e-35), dim=2)

        return prob


class ChimeraACVAE_Encoder(nn.Module):
    def __init__(self, n_freq, n_label):
        super(ChimeraACVAE_Encoder, self).__init__()
        self.type = type
        self.conv1 = SiLUConv2D(n_freq, n_freq//2,
                                (1, 5), (1, 1), (0, 2))
        self.conv2 = SiLUConv2D(n_freq//2, n_freq//4,
                                (1, 4), (1, 2), (0, 1))
        self.conv3z = nn.Conv2d(n_freq//4, n_freq//8*2,
                                (1, 4), (1, 2), (0, 1))
        self.conv3c = SiLUConv2D(n_freq//4, n_freq//4,
                                 (1, 4), (1, 2), (0, 1))
        self.conv4c = nn.Conv2d(n_freq//4, n_label,
                                (1, 4), (1, 2), (0, 1))
        self.softmax = nn.Softmax(dim=1)

        nn.init.normal_(self.conv3z.weight, std=0.01)
        nn.init.zeros_(self.conv3z.bias)
        nn.init.normal_(self.conv4c.weight, std=0.01)
        nn.init.zeros_(self.conv4c.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)

        h = self.conv1(x)
        yz = self.conv2(h)

        z = self.conv3z(yz)
        mu, logvar = z.split(z.size(1)//2, dim=1)

        y = self.conv3c(yz)
        y = self.conv4c(y)
        y = torch.mean(y, dim=3)
        y = self.softmax(y)
        y = torch.squeeze(torch.clamp(y, 1.e-35), dim=2)

        return mu, logvar, y


class ChimeraACVAE_Decoder(nn.Module):
    def __init__(self, n_freq, n_label):
        super(ChimeraACVAE_Decoder, self).__init__()
        self.deconv1 = SiLUDeconv2D(n_freq//8+n_label, n_freq//4,
                                    (1, 4), (1, 2), (0, 1))
        self.deconv2 = SiLUDeconv2D(n_freq//4+n_label, n_freq//2,
                                    (1, 4), (1, 2), (0, 1))
        self.deconv3 = nn.ConvTranspose2d(n_freq//2+n_label, n_freq,
                                          (1, 5), (1, 1), (0, 2))

        nn.init.normal_(self.deconv3.weight, std=0.01)
        nn.init.zeros_(self.deconv3.bias)

    def concat_xy(self, x, y):
        n_h, n_w = x.shape[2:4]
        return torch.cat((
            x, y.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_h, n_w)), dim=1)

    def forward(self, z, c):
        h = self.deconv1(self.concat_xy(z, c))
        h = self.deconv2(self.concat_xy(h, c))
        h = self.deconv3(self.concat_xy(h, c))

        h = h.permute(0, 2, 1, 3)

        return torch.clamp(h, min=-80.)
