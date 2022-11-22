#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# models.py: network architecture
# author: Li Li (lili-0805@ieee.org)
#


import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class CVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(), device=mu.device))

        # Reparameterization trick
        return mu + torch.exp(logvar / 2) * eps

    def gaussian_nll(self, x, mu, logvar, reduce="sum"):
        x_prec = torch.exp(-logvar)
        x_diff = x - mu
        x_power = -.5 * (x_diff * x_diff) * x_prec
        loss = (logvar + math.log(2 * math.pi)) / 2 - x_power
        if reduce == "sum":
            gaussian_nll = torch.sum(loss)
        elif reduce == "mean":
            gaussian_nll = torch.mean(loss)

        return gaussian_nll

    def forward(self, x, c):
        c = c.unsqueeze(0).repeat(x.size(0), 1)
        self.z_mu, self.z_logvar = self.encoder(x, c)
        z = self.sample_z(self.z_mu, self.z_logvar)
        self.x_logvar = self.decoder(z, c)

        return self.x_logvar

    def loss(self, x):
        # closed-form of KL divergence between N(mu, var) and N(0, I)
        kl_loss_el = .5 * (self.z_mu.pow(2) + self.z_logvar.exp()
                           - self.z_logvar - 1)
        kl_loss = torch.sum(kl_loss_el) / x.numel()

        # negative log-likelihood of complex proper Gaussian distribution
        logvar = self.x_logvar - math.log(2.)
        x_zero = torch.zeros(self.x_logvar.size(), device=self.x_logvar.device)

        nll_real = self.gaussian_nll(x, x_zero, logvar, "sum") / x.numel()
        nll_imag = self.gaussian_nll(x_zero, x_zero, logvar, "sum") / x.numel()
        nll_loss = nll_real + nll_imag

        loss = kl_loss + nll_loss

        return loss, kl_loss, nll_loss


class ACVAE(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super(ACVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(), device=mu.device),
                       requires_grad=True)

        return mu + torch.exp(logvar / 2) * eps

    def gaussian_nll(self, x, mu, logvar, reduce="sum"):
        x_prec = torch.exp(-logvar)
        x_diff = x - mu
        x_power = -.5 * (x_diff * x_diff) * x_prec
        loss = (logvar + math.log(2 * math.pi)) / 2 - x_power
        if reduce == "sum":
            gaussian_nll = torch.sum(loss)
        elif reduce == "mean":
            gaussian_nll = torch.mean(loss)

        return gaussian_nll

    def forward(self, x, c):
        c = c.unsqueeze(0).repeat(x.size(0), 1)

        self.z_mu, self.z_logvar = self.encoder(x, c)
        z = self.sample_z(self.z_mu, self.z_logvar)
        self.x_logvar = self.decoder(z, c)
        self.dec_x = torch.exp(self.x_logvar / 2)
        self.x_prob = self.classifier(x)
        self.dec_x_prob = self.classifier(self.dec_x)

        return self.x_logvar

    def loss(self, x, c):
        # closed-form of KL divergence between N(mu, var) and N(0, I)
        kl_loss_el = .5 * (self.z_mu.pow(2) + self.z_logvar.exp()
                           - self.z_logvar - 1)
        kl_loss = torch.sum(kl_loss_el) / x.numel()

        # negative log-likelihood of complex proper Gaussian distribution
        logvar = self.x_logvar - math.log(2.)
        x_zero = torch.zeros(self.x_logvar.size(), device=self.x_logvar.device)

        nll_real = self.gaussian_nll(x, x_zero, logvar, "sum") / x.numel()
        nll_imag = self.gaussian_nll(x_zero, x_zero, logvar, "sum") / x.numel()
        nll_loss = nll_real + nll_imag

        # classification loss
        c = c.unsqueeze(0).repeat(x.size(0), 1)
        cls_loss_x = torch.mean(
            torch.sum(-torch.log(self.x_prob) * c, dim=1))
        cls_loss_dec_x = torch.mean(
            torch.sum(-torch.log(self.dec_x_prob) * c, dim=1))

        loss = kl_loss + nll_loss + 0.5 * cls_loss_x + 0.5 * cls_loss_dec_x

        return loss, kl_loss, nll_loss, cls_loss_x, cls_loss_dec_x


class ChimeraACVAE(nn.Module):
    def __init__(self, encoder, decoder, mvae_encoder=None, mvae_decoder=None):
        super(ChimeraACVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mvae_encoder = mvae_encoder
        self.mvae_decoder = mvae_decoder

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(), device=mu.device),
                       requires_grad=True)

        return mu + torch.exp(logvar / 2) * eps

    def sample_c(self, logits, tau=1):
        c = nn.functional.gumbel_softmax(logits, tau)

        return c

    def gaussian_nll(self, x, mu, logvar, reduce="sum"):
        x_prec = torch.exp(-logvar)
        x_diff = x - mu
        x_power = -.5 * (x_diff * x_diff) * x_prec
        loss = (logvar + math.log(2 * math.pi)) / 2 - x_power
        if reduce == "sum":
            gaussian_nll = torch.sum(loss)
        elif reduce == "mean":
            gaussian_nll = torch.mean(loss)

        return gaussian_nll

    def forward(self, x, c):
        self.z_mu, self.z_logvar, self.x_prob = self.encoder(x)
        z = self.sample_z(self.z_mu, self.z_logvar)
        c = c.unsqueeze(0).repeat(x.size(0), 1)
        self.x_logvar = self.decoder(z, c)
        self.dec_x = torch.exp(self.x_logvar / 2)
        self.dec_x_prob = self.encoder(self.dec_x)[2]

        sampled_c = self.sample_c(self.x_prob)
        self.x_logvar_est_c = self.decoder(z, sampled_c)
        self.dec_x_est_c = torch.exp(self.x_logvar_est_c / 2)
        self.dec_x_est_c_prob = self.encoder(self.dec_x_est_c)[2]

        return None

    def loss(self, x, c):
        # ============ vanilla ACVAE loss ==============
        # closed-form of KL divergence between N(mu, var) and N(0, I)
        kl_loss_el = .5 * (self.z_mu.pow(2) + self.z_logvar.exp()
                           - self.z_logvar - 1)
        kl_loss = torch.sum(kl_loss_el) / x.numel()

        # negative log-likelihood of complex proper Gaussian distribution
        logvar = self.x_logvar - math.log(2.)
        x_zero = torch.zeros(self.x_logvar.size(), device=self.x_logvar.device)
        nll_real = self.gaussian_nll(x, x_zero, logvar, "sum") / x.numel()
        nll_imag = self.gaussian_nll(x_zero, x_zero, logvar, "sum") / x.numel()
        nll_loss = nll_real + nll_imag

        # classification loss
        c = c.unsqueeze(0).repeat(x.size(0), 1)
        cls_loss_x = torch.mean(torch.sum(-torch.log(self.x_prob) * c, dim=1))
        cls_loss_dec_x = torch.mean(torch.sum(
            -torch.log(self.dec_x_prob) * c, dim=1))

        # ============ loss with estimated label ==============
        # reconstruction loss
        logvar_est_c = self.x_logvar_est_c - math.log(2.)
        nll_real_est_c = self.gaussian_nll(
            x, x_zero, logvar_est_c, "sum") / x.numel()
        nll_imag_est_c = self.gaussian_nll(
            x_zero, x_zero, logvar_est_c, "sum") / x.numel()
        nll_loss_est_c = nll_real_est_c + nll_imag_est_c

        # classification loss
        cls_loss_dec_x_est_c = torch.mean(
            torch.sum(-torch.log(self.dec_x_est_c_prob) * c, dim=1))

        # =========== TS loss =============
        # calculate prior distributions using CVAE
        cvae_z_mu, cvae_z_logvar = self.mvae_encoder(x, c)
        cvae_x_logvar = self.mvae_decoder(cvae_z_mu, c)

        # KL divergence between CVAE and ChimeraVAE (Normal distribution)
        # TS_z (Normal distribution)
        cvae_z_dist = torch.distributions.normal.Normal(
            cvae_z_mu, torch.exp(cvae_z_logvar))
        z_dist = torch.distributions.normal.Normal(
            self.z_mu, torch.exp(self.z_logvar))
        ts_z_loss = torch.mean(
            torch.distributions.kl.kl_divergence(cvae_z_dist, z_dist))

        # TS_x (Complex normal distribution)
        lb = x_zero + 1.e-8
        x_var = torch.maximum(torch.exp(self.x_logvar), lb)
        ts_x_loss = torch.mean(
            self.x_logvar - cvae_x_logvar - 1. + \
                torch.exp(cvae_x_logvar) / x_var)

        # losses for reconstructed x with estimated label 
        # (Complex normal distribution)
        x_var_est_c = torch.maximum(torch.exp(self.x_logvar_est_c), lb)
        ts_x_loss_est_c = torch.mean(
            self.x_logvar_est_c - cvae_x_logvar - 1. + \
                torch.exp(cvae_x_logvar) / x_var_est_c)

        # =========== total loss =========
        # optimal weight for ts_z_loss is 10, for others 1
        loss = kl_loss + nll_loss + cls_loss_x + cls_loss_dec_x \
            + nll_loss_est_c + cls_loss_dec_x_est_c \
            + 10 * ts_z_loss + ts_x_loss + ts_x_loss_est_c

        return loss, kl_loss, nll_loss, \
            cls_loss_x, cls_loss_dec_x, \
            nll_loss_est_c, cls_loss_dec_x_est_c, \
            ts_z_loss, ts_x_loss, ts_x_loss_est_c


# ========== for separation phase ==========
class SourceModel(nn.Module):
    def __init__(self, decoder, z, c):
        super(SourceModel, self).__init__()
        self.decoder = decoder
        self.z_layer = z
        self.c_layer = c
        self.softmax = nn.Softmax(dim=1)

    def gaussian_nll(self, x, mu, logvar, reduce="sum"):
        x_prec = torch.exp(-logvar)
        x_diff = x - mu
        x_power = -.5 * (x_diff * x_diff) * x_prec
        loss = (logvar + math.log(2 * math.pi)) / 2 - x_power
        if reduce == "sum":
            gaussian_nll = torch.sum(loss)
        elif reduce == "mean":
            gaussian_nll = torch.mean(loss)

        return gaussian_nll

    def forward(self):
        label = self.softmax(self.c_layer)
        self.x_logvar = self.decoder(self.z_layer, label)

        return None

    def loss(self, x):
        self.forward()
        logvar = self.x_logvar - math.log(2.)
        x_zero = torch.zeros(self.x_logvar.size(),
                             device=self.x_logvar.device,
                             dtype=torch.float)
        z_zero = torch.zeros(self.z_layer.size(),
                             device=self.z_layer.device,
                             dtype=torch.float)

        kl_loss = self.gaussian_nll(
            self.z_layer, z_zero, z_zero, "sum") / x.numel()
        nll_real = self.gaussian_nll(
            x, x_zero, logvar, "sum") / x.numel()
        nll_imag = self.gaussian_nll(
            x_zero, x_zero, logvar, "sum") / x.numel()
        nll_loss = nll_real + nll_imag

        loss = nll_loss + kl_loss

        return loss

    def get_power_spec(self, cpu=True):
        label = self.softmax(self.c_layer)
        if cpu is True:
            return np.squeeze(np.exp(self.decoder(
                self.z_layer, label).detach().to("cpu").numpy()), axis=1)
        else:
            return torch.squeeze(torch.exp(self.decoder(
                self.z_layer, label)), dim=1)

    def get_label(self, cpu=True):
        label = self.softmax(self.c_layer)
        if cpu is True:
            return label.detach().to("cpu").numpy()
        else:
            return label
