# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FastMVAE.py
# author: Li Li (lili-0805@ieee.org)
#


import numpy as np
import torch
from mvae_ss.setting.constant import EPSI, USE_NORM, ITERATIONS
from mvae_ss.nn.functions import construct_model
from mvae_ss.utils.bss import local_normalize, update_w


def FastMVAE(X, config):
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
    encoder, decoder, classifier = module[0], module[1], module[2]
    checkpoint = torch.load(config.model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # n_ch x n_freq x n_frame
    Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)
    labels = torch.eye(config.label_dim, dtype=torch.float, device=device)

    # Algorithm for FastMVAE
    # Iterative update
    print('Iteration:      ')
    for it in range(ITERATIONS):
        print(f'\rIteration: {it}', end="")

        gv = np.mean(np.power(Y_abs[:, 1:I, :], 2), axis=(1, 2), keepdims=True)
        Y_abs_array_norm = Y_abs / np.sqrt(gv)
        for n in range(N):
            y_abs = torch.from_numpy(np.asarray(
                Y_abs_array_norm[n, None, None, 1:, :],
                dtype="float32")).to(device)

            # update label
            chat = classifier(y_abs)
            cls_idx = torch.argmax(chat, dim=1)
            if config.label_type == "onehot":
                chat = labels[cls_idx]

            # update z
            z_mu, z_logvar = encoder(y_abs, chat)

            # update logvar
            beta = torch.exp(-z_logvar)
            if config.alpha == "mean":
                alpha = torch.mean(beta, keepdim=True)
            else:
                alpha = torch.ones_like(z_logvar) * config.alpha
            zhat = beta / (beta + alpha) * z_mu
            logvar = decoder(zhat, chat)

            Q[n, 1:I, :] = np.squeeze(np.exp(
                logvar.detach().to("cpu").numpy()), axis=1)

        Q = np.maximum(Q, EPSI)
        gv = np.mean(np.divide(
            P[:, 1:I, :], Q[:, 1:I, :]), axis=(1, 2), keepdims=True)
        Rhat = np.multiply(Q, gv)
        Rhat[:, 0, :] = R[:, 0, :]
        R = Rhat.transpose(1, 2, 0)

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
