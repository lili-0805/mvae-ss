#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# functions.py: functions for all network architectures
# author: Li Li (lili-0805@ieee.org)
#


import torch
import os
import mvae_ss.nn.modules as modules
import mvae_ss.nn.models as models


def construct_model(config, mode="train"):
    assert mode in ["train", "test"], "Mode should be train or test!"
    if config.source_model == "CVAE":
        encoder = modules.CVAE_Encoder(config.nn_freq, config.n_src)
        decoder = modules.CVAE_Decoder(config.nn_freq, config.n_src)
        model = models.CVAE(encoder, decoder)
        module = [encoder, decoder]

    elif config.source_model == "ACVAE":
        encoder = modules.CVAE_Encoder(config.nn_freq, config.n_src)
        decoder = modules.CVAE_Decoder(config.nn_freq, config.n_src)
        classifier = modules.ACVAE_Classifier(config.nn_freq, config.n_src)
        model = models.ACVAE(encoder, decoder, classifier)
        module = [encoder, decoder, classifier]

    elif config.source_model == "ChimeraACVAE":
        encoder = modules.ChimeraACVAE_Encoder(config.nn_freq, config.n_src)
        decoder = modules.ChimeraACVAE_Decoder(config.nn_freq, config.n_src)

        if mode == "train":
            cvae_encoder = modules.CVAE_Encoder(config.nn_freq, config.n_src)
            cvae_decoder = modules.CVAE_Decoder(config.nn_freq, config.n_src)
            teacher_model = config.teacher_model
            checkpoint = torch.load(teacher_model)
            cvae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            cvae_decoder.load_state_dict(checkpoint['decoder_state_dict'])
            for para in cvae_encoder.parameters():
                para.requires_grad = False
            for para in cvae_decoder.parameters():
                para.requires_grad = False

            model = models.ChimeraACVAE(encoder, decoder,
                                        cvae_encoder, cvae_decoder)

        elif mode == "test":
            model = models.ChimeraACVAE(encoder, decoder)

        module = [encoder, decoder]

    else:
        raise ValueError("Wrong source model type!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.cuda(device)

    if mode == "train":
        return model, device
    elif mode == "test":
        return module, device


def set_optimizer(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return optimizer


def load_pretrained_model(model, optimizer, config):
    checkpoint = torch.load(config.pretrained_model)

    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config.source_model == "ACVAE":
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    return start_epoch


def set_cudnn():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    return None


def snapshot(epoch, model, optimizer, config):
    print(f"Saving the model at {epoch} epoch...", end="")
    if config.source_model in ['CVAE', 'ChimeraACVAE']:
        torch.save({'epoch': epoch,
                    'encoder_state_dict': model.encoder.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(config.save_model_path, f'{epoch}.model'))

    elif config.source_model in ['ACVAE']:
        torch.save({'epoch': epoch,
                    'encoder_state_dict': model.encoder.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'classifier_state_dict': model.classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(config.save_model_path, f'{epoch}.model'))

    else:
        raise ValueError("Wrong source model type!")

    print("Done!")

    return None


def write_log(writer, losses, total_iteration, config):
    if config.source_model == "CVAE":
        loss_name = ["Loss/total loss", "CVAE Loss/KL", "CVAE Loss/nll"]
    elif config.source_model == "ACVAE":
        loss_name = ["Loss/total loss", "ACVAE Loss/KL",
                     "ACVAE Loss/nll", "ACVAE Loss/cls_x",
                     "ACVAE Loss/cls_dec"]
    elif config.source_model == "ChimeraACVAE":
        loss_name = ["Loss/total loss", "ACVAE Loss/KL",
                     "ACVAE Loss/nll", "ACVAE Loss/cls_x",
                     "ACVAE Loss/cls_dec", "estimated label/rec",
                     "estimated label/cls_dec", "TS loss/TS_z",
                     "TS loss/TS_x", "TS loss/TS_x_est_label"]
    else:
        raise ValueError("Wrong source model!")

    for k in range(len(losses)):
        writer.add_scalar(loss_name[k], losses[k], total_iteration)

    return None
