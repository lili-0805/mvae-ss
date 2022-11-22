# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# utils.py: utility functions
# author: Li Li (lili-0805@ieee.org)
#


import os
import datetime
from mvae_ss.setting.constant import SAMP_RATE, RATIO
from mvae_ss.setting.constant import WIN_TYPE, WIN_LEN, WIN_SHIFT
from mvae_ss.setting.constant import USE_NORM, ITERATIONS


class Logger:
    def __init__(self, logf, add=True):
        if add and os.path.isfile(logf):
            self.out = open(logf, 'a')
            self.out.write(f"\n{datetime.datetime.now()}\n")
        else:
            if os.path.isfile(logf):
                os.remove(logf)
            self.out = open(logf, 'a')
            self.out.write(f"{datetime.datetime.now()}\n")

    def __del__(self):
        if self.out is not None:
            self.close()

    def __call__(self, msg):
        print(msg)
        self.out.write(f"{msg}\n")
        self.out.flush()

    def close(self):
        self.out.close()
        self.out = None


def generate_test_log(config):
    logprint = Logger(f"{config.output_dir}log.txt", add=False)
    logprint("Data")
    logprint(f"\tInput folder: {config.input_dir}")
    logprint(f"\tOutput folder: {config.output_dir}")
    logprint("\nSTFT settings")
    logprint(f"\tSampling rate: {SAMP_RATE}")
    logprint(f"\tWindow: {WIN_TYPE} with window length {WIN_LEN}"
             f" and shift {WIN_SHIFT}")
    logprint(f"\tFrame #: {RATIO}x")
    logprint("\nSeparation settings")
    logprint(f"\tUse normalization: {USE_NORM}")
    logprint(f"\tIteration #: {ITERATIONS}")
    logprint("\nNetwork")
    logprint(f"\tNetwork model: {config.model_path}")
    logprint(f"\tDimension of label: {config.label_dim}")
    logprint("\nMethod")
    logprint(f"\tAlgorithm: {config.algorithm}")
    logprint(f"\tSource model: {config.source_model}")
    if config.algorithm == "FastMVAE":
        logprint(f"\tType of conditional variable: {config.label_type}")
        logprint(f"\tWeight for PoE estimation: {config.alpha}")

    return None


def str2bool(x):
    if x.lower()[0] == "t":
        return True
    elif x.lower()[0] == "f":
        return False


def num2str(n):
    if n < 10:
        return f'00{n}'
    elif n < 100:
        return f'0{n}'
    elif n < 1000:
        return f'{n}'
