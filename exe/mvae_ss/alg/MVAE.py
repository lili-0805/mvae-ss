# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MVAE.py: MVAE algorithm
# author: Li Li (lili-0805@ieee.org)
#


import torch
import numpy as np
from mvae_ss.setting.constant import EPSI, USE_NORM, ITERATIONS
from mvae_ss.nn.models import SourceModel
from mvae_ss.nn.functions import construct_model
from mvae_ss.utils.bss import local_normalize, update_w


def MVAE(X, config):
    # check errors and set default values
    I, J, M = X.shape
    N = M
    if N > I:
        raise ValueError("The input spectrogram might be wrong."
                         "The size of it must be (freq x frame x ch).")

    # Initialization
    W = np.zeros((I, M, N), dtype=np.complex)
    for i in range(I):
        W[i, :, :] = np.eye(N)
    Y = X @ W.conj()

    P = np.maximum(np.abs(Y) ** 2, EPSI)
    R = P.copy()
    if USE_NORM:
        W, R, P = local_normalize(W, R, P, I, J)
    P = P.transpose(2, 0, 1)
    R = R.transpose(2, 0, 1)
    Q = np.zeros((N, I, J))

    # load trained networks
    module, device = construct_model(config, mode="test")
    encoder, decoder = module[0], module[1]
    checkpoint = torch.load(config.model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # initial z and c
    Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)
    gv = np.mean(np.power(Y_abs[:, 1:, :], 2), axis=(1, 2), keepdims=True)
    Y_abs_norm = Y_abs / np.sqrt(gv)
    eps = np.ones(Y_abs_norm.shape) * EPSI
    Y_abs_array_norm = np.maximum(Y_abs_norm, eps)[:, None]

    zs, cs, models, optims = [], [], [], []
    for n in range(N):
        y_abs = torch.from_numpy(np.asarray(
            Y_abs_array_norm[n, None, :, 1:, :], dtype="float32")).to(device)
        label = torch.from_numpy(
            np.ones((1, config.label_dim),
                    dtype="float32") / config.label_dim).to(device)
        z = encoder(y_abs, label)[0]
        zs.append(z)
        cs.append(label)
        Q[n, 1:, :] = np.squeeze(
            np.exp(decoder(z, label).detach().to("cpu").numpy()), axis=1)

    Q = np.maximum(Q, EPSI)
    gv = np.mean(np.divide(P[:, 1:, :], Q[:, 1:, :]), axis=(1, 2),
                 keepdims=True)
    Rhat = np.multiply(Q, gv)
    Rhat[:, 0, :] = R[:, 0, :]
    R = Rhat

    # Model construction
    for para in decoder.parameters():
        para.requires_grad = False

    for n in range(N):
        z_para = torch.nn.Parameter(zs[n].type(torch.float),
                                    requires_grad=True)
        c_para = torch.nn.Parameter(cs[n].type(torch.float),
                                    requires_grad=True)
        src_model = SourceModel(decoder, z_para, c_para)
        if device == "cuda":
            src_model.cuda(device)
        optimizer = torch.optim.Adam(src_model.parameters(), lr=0.01)
        models.append(src_model)
        optims.append(optimizer)

    # initialize z, c by running BP 100 iterations
    Q = np.zeros((N, I, J))
    for n in range(N):
        y_abs = torch.from_numpy(np.asarray(
            Y_abs_array_norm[n, None, :, 1:, :], dtype="float32")).to(device)
        for _ in range(100):
            optims[n].zero_grad()
            loss = models[n].loss(y_abs)
            loss.backward()
            optims[n].step()
        Q[n, 1:I, :] = models[n].get_power_spec(cpu=True)

    Q = np.maximum(Q, EPSI)
    gv = np.mean(np.divide(P[:, 1:I, :], Q[:, 1:I, :]), axis=(1, 2),
                 keepdims=True)
    Rhat = np.multiply(Q, gv)
    Rhat[:, 0, :] = R[:, 0, :]
    R = Rhat

    # Algorithm for MVAE
    # Iterative update
    for it in range(ITERATIONS):
        print(f'\rIteration: {it}', end="")

        Y_abs_array_norm = Y_abs / np.sqrt(gv)
        y_abs = torch.from_numpy(np.asarray(Y_abs_array_norm[:, None, 1:, :],
                                            dtype="float32")).to(device)
        for n in range(N):
            for _ in range(100):
                optims[n].zero_grad()
                loss = models[n].loss(y_abs[n, None])
                loss.backward()
                optims[n].step()
            Q[n, 1:, :] = models[n].get_power_spec(cpu=True)
        Q = np.maximum(Q, EPSI)
        gv = np.mean(np.divide(P[:, 1:, :], Q[:, 1:, :]), axis=(1, 2),
                     keepdims=True)
        Rhat = np.multiply(Q, gv)
        Rhat[:, 0, :] = R[:, 0, :]
        R = Rhat.transpose(1, 2, 0)

        # update W
        W = update_w(X, R, W)
        Y = X @ W.conj()
        Y_abs = np.abs(Y)
        Y_pow = np.power(Y_abs, 2)
        P = np.maximum(Y_pow, EPSI)

        if USE_NORM:
            W, R, P = local_normalize(W, R, P, I, J)

        Y_abs = Y_abs.transpose(2, 0, 1)
        P = P.transpose(2, 0, 1)
        R = R.transpose(2, 0, 1)

    return Y, W
