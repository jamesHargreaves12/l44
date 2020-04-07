# -*- coding: utf-8 -*-
# Code From:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.htmlhttps://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# THis tutorial uses alot of the best practices from the original paper

from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
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
    plot_real_vs_fake

if __name__ == "__main__":

    # Root directory for dataset
    cfg = yaml.load(open("config_aegan.yaml"))

    device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

    dataloader = get_dataset(shuffle=False)
    labs = open('data/FERlabs.txt', 'r').read().split(',')

    netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)
    netE, optimizerE = get_model_and_optimizer(VariationalEncoder, cfg["enc_path"], cfg)

    emotion_latents = defaultdict(list)
    lab_iter = iter(labs)
    output_file = open("data/latent_to_emotion.csv", "w+")
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            X = data[0].to(device)
            Z_mu, _ = netE(X)
            for z in Z_mu.cpu().numpy():
                lab = next(lab_iter)
                emotion_latents[lab].append(z)
                output_file.write(",".join([str(x) for x in z]) + "," + lab + "\n")
    output_file.close()



    average_emotion = {}
    for emotion in emotion_latents.keys():
        df_mu = pd.DataFrame([x for x in emotion_latents[emotion]])
        df_mu_average = df_mu.mean().to_numpy()
        average_emotion[emotion] = df_mu_average

    # Trying to find the average person with each emotion - poor results:
    # with torch.no_grad():
    #     for emotion in average_emotion.keys():
    #         input_mu = torch.from_numpy(average_emotion[emotion][0])
    #         input_logvar = torch.from_numpy(average_emotion[emotion][1])
    #         reshape = input_mu.reshape([1, cfg['nz'], 1, 1]).float()
    #         fake = netG(reshape)
    #         img = vutils.make_grid(fake, padding=5, normalize=True).cpu()
    #         plt.imshow(img[0], cmap=cm.gray)
    #         plt.savefig("output_images/emotion_{}.png".format(emotion))

    # I think the issue is that this network contains batch normalisation layer
    num_attempts = 8
    results = []
    print("Generating changed emotions")
    test_batch = next(iter(dataloader))
    X = test_batch[0].to(device)
    with torch.no_grad():
        Z_mus, _ = netE(X)
        fakes_no_change = netG(Z_mus.reshape(-1, cfg['nz'], 1, 1))
        fake_imgs = vutils.make_grid(fakes_no_change, padding=2, normalize=True)[:64]
        plot_real_vs_fake(X, fake_imgs, show=True)

        for attempt in tqdm(range(num_attempts)):
            test_label = labs[attempt]
            result = {"7": X[attempt]}
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
    f, axarr = plt.subplots(num_attempts, 8)
    for i, result in enumerate(results):
        for em in result.keys():
            img = vutils.make_grid(result[em], padding=2, normalize=True).cpu()
            axarr[i, int(em)].imshow(np.transpose(img, (1, 2, 0)))
        plt.savefig("output_images/emotion_change.png")
