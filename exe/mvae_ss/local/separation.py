# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# separation.py
# auther: Li Li (lili-0805@ieee.org)
#


import os
import argparse
import numpy as np
import mvae_ss.utils.data as data
from scipy.io import wavfile
from scipy import signal
from mvae_ss.alg.MVAE import MVAE
from mvae_ss.alg.FastMVAE import FastMVAE
from mvae_ss.alg.FastMVAE2 import FastMVAE2
from mvae_ss.utils.utils import generate_test_log
from mvae_ss.utils.bss import back_projection
from mvae_ss.setting.constant import SAMP_RATE, RATIO
from mvae_ss.setting.constant import WIN_TYPE, WIN_LEN, WIN_SHIFT
from mvae_ss.setting.constant import REF_MIC, N_FREQ


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument('--algorithm', type=str, required=True,
                        help="source separation algorithm",
                        choices=["MVAE", "FastMVAE", "FastMVAE2"])
    parser.add_argument('--source_model', type=str, required=True,
                        help="corresponding source model",
                        choices=["CVAE", "ACVAE", "ChimeraACVAE"])
    parser.add_argument('--input_dir', type=str, required=True,
                        help="path for input test data")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="path for output data")
    parser.add_argument('--model_path', type=str, required=True,
                        help="path for a trained model")
    parser.add_argument('--label_dim', type=int, required=True,
                        help="number of sources in training dataset")
    parser.add_argument('--label_type', type=str,
                        help="label_type for FastMVAE",
                        choices=["cont", "onehot"], default="cont")
    parser.add_argument('--alpha', type=float,
                        help="prior weight in PoE-based FastMVAE")
    config = parser.parse_args()
    config.nn_freq = N_FREQ - 1
    config.n_src = config.label_dim
    generate_test_log(config)

    # ======== main process ========
    input_dir_names = sorted(os.listdir(config.input_dir))

    for folder in input_dir_names:
        print(config.input_dir + folder)
        save_path = os.path.join(config.output_dir, folder)
        os.makedirs(save_path, exist_ok=True)

        # Input data and resample
        mix = data.load_wav(os.path.join(config.input_dir, folder), SAMP_RATE)
        ns = mix.shape[1]

        # STFT
        frames_ = np.floor((mix.shape[0] + 2*WIN_SHIFT) / WIN_SHIFT)
        frames = int(np.ceil(frames_ / RATIO) * RATIO)

        X = np.zeros((int(WIN_LEN / 2 + 1), int(frames), mix.shape[1]),
                     dtype='complex')
        for n in range(mix.shape[1]):
            f, t, X[:, :int(frames_), n] = signal.stft(
                mix[:, n], nperseg=WIN_LEN, window=WIN_TYPE,
                noverlap=WIN_LEN-WIN_SHIFT)

        # source separation
        if config.algorithm == "MVAE":
            Y, W = MVAE(X, config)
        elif config.algorithm == "FastMVAE":
            Y, W = FastMVAE(X, config)
        elif config.algorithm == "FastMVAE2":
            Y, W = FastMVAE2(X, config)

        # back projection
        XbP = np.zeros((X.shape[0], X.shape[1], 1), dtype='complex')
        XbP[:, :, 0] = X[:, :, REF_MIC]
        Z = back_projection(Y, XbP)

        # iSTFT
        sep = np.zeros((WIN_SHIFT * (frames - 1), ns))
        for n in range(ns):
            sep_ = signal.istft(Z[:, :, n], window=WIN_TYPE)[1]
            n_sample = np.minimum(sep_.shape[0], WIN_SHIFT * (frames - 1))
            sep[:n_sample, n] = sep_[:n_sample]

        # normalize wav data
        for n in range(ns):
            sep[:, n] = sep[:, n] / max(abs(sep[:, n])) * 30000

        # save wav files
        for n in range(ns):
            wavfile.write(os.path.join(save_path, f'estimated_signal{n}.wav'),
                          SAMP_RATE, sep[:, n].astype(np.int16))
        np.save(os.path.join(save_path, 'W.npy'), W)
        print(f'\nSeparated signals are saved in {config.output_dir}{folder}')
