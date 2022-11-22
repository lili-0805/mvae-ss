# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# constant.py: constant values
# Author: Li Li (lili-0805@ieee.org)
#


import sys


EPSI = sys.float_info.epsilon

# signal processing
SAMP_RATE = 16000
WIN_TYPE = "hamming"
WIN_LEN = 2048
WIN_SHIFT = 1024
RATIO = 4
N_FREQ = WIN_LEN // 2 + 1

# separation algorithm
REF_MIC = 0
USE_NORM = True
ITERATIONS = 60

# network training
SEG_LEN = 128
MAX_BATCH_SIZE = 16
