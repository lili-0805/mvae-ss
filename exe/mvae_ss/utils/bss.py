# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# bss.py: utility functions for bss
# author: Li Li (lili-0805@ieee.org)


import numpy as np
import numpy.linalg as LA
from mvae_ss.setting.constant import EPSI


# separation
def back_projection(Y, X):
    I, J, M = Y.shape

    if X.shape[2] == 1:
        A = np.zeros((1, M, I), dtype=np.complex)
        Z = np.zeros((I, J, M), dtype=np.complex)
        for i in range(I):
            Yi = np.squeeze(Y[i, :, :]).T  # channels x frames (M x J)
            Yic = np.conjugate(Yi.T)
            A[0, :, i] = X[i, :, 0] @ Yic @ np.linalg.inv(Yi @ Yic)

        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        for m in range(M):
            for i in range(I):
                Z[i, :, m] = A[0, m, i] * Y[i, :, m]

    elif X.shape[2] == M:
        A = np.zeros(M, M, I)
        Z = np.zeros(I, J, M, M)
        for i in range(I):
            for m in range(M):
                Yi = np.squeeze(Y[i, :, :]).T
                Yic = np.conjugate(Yi.T)
                A[0, :, i] = X[i, :, m] @ Yic @ np.linalg.inv(Yi @ Yic)
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        for n in range(M):
            for m in range(M):
                for i in range(I):
                    Z[i, :, n, m] = A[m, n, i] * Y[i, :, n]

    else:
        print('The number of channels in X must be 1 or equal to that in Y.')

    return Z


def local_normalize(W, R, P, I, J, *args):
    lamb = np.sqrt(np.sum(np.sum(P, axis=0), axis=0) / (I * J))

    W = W / np.squeeze(lamb)
    lambPow = lamb ** 2
    P = P / lambPow
    R = R / lambPow
    if len(args) == 1:
        T = args[0]
        T = T / lambPow
        return W, R, P, T
    elif len(args) == 0:
        return W, R, P


def update_w(s, r, w):
    K = w.shape[-1]
    _, N, M = s.shape
    sigma = np.einsum('fnp,fnl,fnq->flpq', s, 1 / r, s.conj())
    sigma /= N
    for k in range(K):
        w[..., k] = LA.solve(
            w.swapaxes(-2, -1).conj() @ sigma[:, k, ...],
            np.eye(K)[None, :, k])
        den = np.einsum(
            'fp,fpq,fq->f',
            w[..., k].conj(), sigma[:, k, ...], w[..., k])
        w[..., k] /= np.maximum(np.sqrt(np.abs(den))[:, None], 1.e-8)
    w += EPSI * np.eye(M)

    return w
