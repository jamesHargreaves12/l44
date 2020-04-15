# -*- coding: utf-8 -*-
# Code From:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.htmlhttps://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# THis tutorial uses alot of the best practices from the original paper

from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import sys

import matplotlib.cm as cm

from collections import defaultdict
from time import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
from IPython.display import HTML
from tqdm import tqdm

from models import Encoder, Generator, Discriminator, initialise, VariationalEncoder
from utils import get_dataset, get_model_and_optimizer, save_images, reparameterize, loss_function_kld, \
    plot_real_vs_fake, get_dataset_celeba


def get_lab(labs, id):
    return np.array(labs[labs["image_id"] == id]["lab"])[0]


def get_lab_df(filepath):
    if 'list_attr_celeba.csv' in filepath:
        lab_df = pd.read_csv(filepath)
        lab_df['lab'] = lab_df["Smiling"].map(lambda x: "Smiling" if x == 1 else "Not Smiling")
    elif 'output_classif.csv' in filepath:
        lab_df = pd.read_csv(filepath, header=None)
        lab_df = lab_df.rename(
            columns={0: "image_id", 1: "Angry", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sad", 6: "Surprise",
                     7: "Neutral"})
        # Could add filter based on confidence of prediction here
        expression_order = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        lab_df['lab'] = lab_df[expression_order].idxmax(axis=1)
    elif 'ExpW' in filepath:
        lab_df = pd.read_csv(filepath, header=None)
        expression_order = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                            'unknown']
        lab_df = lab_df.rename(
            columns={0: "image_id", 1: "size", 2: "usage", 3: 'neutral', 4: 'happiness', 5: 'surprise', 6: 'sadness',
                     7: 'anger', 8: 'disgust', 9: 'fear', 10: 'contempt', 11: 'unknown'})
        lab_df['lab'] = lab_df[expression_order].idxmax(axis=1)

    return lab_df[["image_id", "lab"]]


if __name__ == "__main__":
    cfg = yaml.load(open("config_expw.yaml"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataset(cfg, shuffle=False)
    lab_df = get_lab_df(cfg['label_csv_path'])

    netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)
    netE, optimizerE = get_model_and_optimizer(VariationalEncoder, cfg["enc_path"], cfg)

    emotion_latents = defaultdict(list)
    # lab_iter = iter(labs)
    # output_file = open("data/latent_to_emotion.csv", "w+")
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            if not torch.cuda.is_available() and i > 500 and cfg["speedup_emot_change"]:
                break
            X = data[0].to(device)
            filenames = data[2]
            Z_mu, _ = netE(X)
            for z, fname in zip(Z_mu.cpu().numpy(), filenames):
                lab = get_lab(lab_df, fname)
                emotion_latents[lab].append(z)
                # output_file.write(",".join([str(x) for x in z]) + "," + lab + "\n")
    # output_file.close()

    average_emotion = {}
    for emotion in emotion_latents.keys():
        df_mu = pd.DataFrame([x for x in emotion_latents[emotion]])
        df_mu_average = df_mu.mean().to_numpy()
        average_emotion[emotion] = df_mu_average

    # Trying to find the average person with each emotion - poor results:
    with torch.no_grad():
        for emotion in average_emotion.keys():
            input_mu = torch.from_numpy(average_emotion[emotion])
            reshape = input_mu.reshape([1, cfg['nz'], 1, 1]).float()
            fake = netG(reshape)
            img = vutils.make_grid(fake, padding=5, normalize=True).cpu()
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.savefig("output_images/emotion_{}.png".format(emotion))

    # I think the issue is that this network contains batch normalisation layer
    num_attempts = 8
    results = []
    print("Generating changed emotions")
    test_batch = next(iter(dataloader))
    X = test_batch[0].to(device)
    with torch.no_grad():
        Z_mus, _ = netE(X)
        fakes_no_change = netG(Z_mus.reshape(-1, cfg['nz'], 1, 1))
        # fake_imgs = vutils.make_grid(fakes_no_change, padding=2, normalize=True)[:64]
        # plot_real_vs_fake(X, fake_imgs, show=True)

        for attempt in tqdm(range(num_attempts)):
            fname = test_batch[2][attempt]
            test_label = get_lab(lab_df, fname)
            result = {"Original": X[attempt]}
            Z_mu_np = Z_mus.cpu().numpy()
            for emotion in average_emotion.keys():
                if emotion == test_label:
                    # fake = fakes_no_change[attempt]
                    fake = netG(Z_mus.reshape(-1, cfg['nz'], 1, 1))[attempt]
                    result[emotion] = fake
                else:
                    Z_mu_changed = Z_mu_np.copy()
                    Z_mu_changed[attempt] = Z_mu_np[attempt] - average_emotion[test_label] + average_emotion[emotion]

                    fake = netG(torch.from_numpy(Z_mu_changed).reshape(-1, cfg['nz'], 1, 1).float().to(device))[attempt]
                    result[emotion] = fake
            results.append(result)
    #
    fig, axarr = plt.subplots(num_attempts, len(results[0].keys()), constrained_layout=False)
    keys = list(results[0].keys())
    for i, k in enumerate(keys):
        axarr[0, i].title.set_text(k)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i, result in enumerate(results):
        for em in result.keys():
            img = vutils.make_grid(result[em], padding=2, normalize=True).cpu()
            axarr[i, keys.index(em)].imshow(np.transpose(img, (1, 2, 0)))
            axarr[i, keys.index(em)].set_xticklabels([])
            axarr[i, keys.index(em)].set_yticklabels([])
        plt.savefig("output_images/emotion_change.png")
