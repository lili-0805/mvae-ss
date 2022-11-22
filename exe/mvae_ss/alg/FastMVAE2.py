# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FastMVAE2.py
# author: Li Li (lili-0805@ieee.org)
#


import torch
import numpy as np
from mvae_ss.nn.functions import construct_model
from mvae_ss.setting.constant import EPSI, USE_NORM, ITERATIONS
from mvae_ss.utils.bss import update_w, local_normalize


def FastMVAE2(X, config):
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
    encoder.eval()
    decoder.eval()

    # n_ch x n_freq x n_frame
    Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)

    # Algorithm for FastMVAE2
    # Iterative update
    for it in range(ITERATIONS):
        print(f'\rIteration: {it}', end="")

        gv = np.mean(np.power(Y_abs[:, 1:I, :], 2), axis=(1, 2), keepdims=True)
        Y_abs_array_norm = Y_abs / np.sqrt(gv)
        y_abs = torch.from_numpy(np.asarray(Y_abs_array_norm[:, None, 1:, :],
                                            dtype="float32")).to(device)

        # update label and latent variable
        z_mu, _, chat = encoder(y_abs)
        # update logvar
        logvar = decoder(z_mu, chat)

        Q[:, 1:I, :] = np.squeeze(
            np.exp(logvar.detach().to("cpu").numpy()), axis=1)
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
