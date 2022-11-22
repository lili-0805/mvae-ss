# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# train_source_model.py: Training script
# author: Li Li (lili-0805@ieee.org)
#

import os
import math
import argparse
import torch
import numpy as np
import mvae_ss.utils.data as data
import mvae_ss.nn.functions as functions
from mvae_ss.setting.constant import SEG_LEN, N_FREQ, MAX_BATCH_SIZE
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training source model")

    parser.add_argument('--source_model', type=str,
                        help="source model type",
                        choices=['CVAE', 'ACVAE', 'ChimeraACVAE'])
    parser.add_argument('--train_data', type=str,
                        help="training dataset")
    parser.add_argument('--save_model_path', type=str,
                        help="path for saving model")
    parser.add_argument('--teacher_model', type=str,
                        help="path for CVAE teacher model", default=None)

    parser.add_argument('--epoch', type=int,
                        help="# of epochs for training")
    parser.add_argument('--snapshot', type=int,
                        help="interval of snapshot")
    parser.add_argument('--iteration', type=int,
                        help="number of iterations")
    parser.add_argument('--learning_rate', type=float,
                        help="learning rate")
    parser.add_argument('--pretrained_model', type=str,
                        help="pretrained model", default=None)

    config = parser.parse_args()
    writer = SummaryWriter(config.save_model_path)

    # Constant values
    n_epoch = config.epoch
    n_iter = config.iteration

    # =============== Directories and data ===============
    # Set input directories and data paths
    save_path = config.save_model_path

    src_folders = sorted(os.listdir(config.train_data))
    data_paths = [f"{config.train_data}{f}/cspec/" for f in src_folders]
    label_paths = [f"{config.train_data}{f}/label.npy" for f in src_folders]
    stat_paths = [f"{config.train_data}{f}/train_cspecstat.npy"
                  for f in src_folders]
    config.n_src = len(src_folders)

    src_data = [sorted(os.listdir(p)) for p in data_paths]
    n_src_data = [len(d) for d in src_data]
    src_bsize = [math.floor(n) // n_iter for n in n_src_data]
    labels = [np.load(p) for p in label_paths]

    # =============== Set model ==============
    config.nn_freq = N_FREQ - 1
    model, device = functions.construct_model(config)
    optimizer = functions.set_optimizer(model, config)
    # load pretrained model
    if os.path.isfile(config.pretrained_model):
        start_epoch = functions.load_pretrained_model(model, optimizer, config)
    elif config.pretrained_model == "" or config.pretrained_model is None:
        start_epoch = 1
    else:
        start_epoch = 1
        writer.add_text("log", "The pretrained model does not exist.")
    # set cudnn
    functions.set_cudnn()

    # =============== Train model ===============
    try:
        for epoch in range(start_epoch, n_epoch+1):
            perms = [np.random.permutation(n) for n in n_src_data]
            perms_data = []

            for i in range(config.n_src):
                perms_data.append([src_data[i][j] for j in perms[i]])

            for i in range(n_iter):
                for j in range(config.n_src):
                    x = data.dat_load_trunc(
                        perms_data[j][i*src_bsize[j]:(i+1)*src_bsize[j]],
                        data_paths[j], SEG_LEN, MAX_BATCH_SIZE)[0]
                    x = data.prenorm(stat_paths[j], x)[0]
                    mag_x = np.linalg.norm(x, axis=1, keepdims=True)
                    x = torch.from_numpy(
                        np.asarray(mag_x, dtype="float32")).to(device)
                    c = torch.from_numpy(
                        np.asarray(labels[j], dtype="float32")).to(device)

                    # update trainable parameters
                    optimizer.zero_grad()
                    model(x, c)
                    if config.source_model == "CVAE":
                        losses = model.loss(x)
                    elif config.source_model in ["ACVAE", "ChimeraACVAE"]:
                        losses = model.loss(x, c)
                    losses[0].backward()
                    optimizer.step()

                    # write log
                    total_iteration = (epoch-1)*(n_iter*config.n_src) \
                        + i*config.n_src + j
                    functions.write_log(writer, losses,
                                        total_iteration, config)

            if epoch % config.snapshot == 0:
                functions.snapshot(epoch, model, optimizer, config)

    except KeyboardInterrupt:
        writer.add_text('error', "\nKeyboard interrupt, exit.")

    else:
        writer.add_text('success', "Training done!")

    finally:
        writer.add_text('model', f"Output: {config.save_model_path}")
        writer.close()
