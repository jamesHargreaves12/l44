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

from models import Encoder, Generator, Discriminator, initialise, VariationalEncoder
from utils import get_dataset, get_model_and_optimizer, save_images, reparameterize, loss_function_kld, \
    plot_real_vs_fake

if __name__ == "__main__":

    # Root directory for dataset
    cfg = yaml.load(open("config_aegan.yaml"))

    device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

    dataloader = get_dataset(1, False)
    df = pd.read_csv('data/FER_orig/train.csv')
    labs = df["emotion"]

    netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)
    netE, optimizerE = get_model_and_optimizer(VariationalEncoder, cfg["enc_path"], cfg)

    criterion = nn.BCELoss()

    calc_BCE_loss = nn.BCELoss()
    calc_MSE_loss = nn.MSELoss()

    emotion_latents = defaultdict(list)

    with torch.no_grad():
        for i, (data, lab) in enumerate(zip(dataloader, labs), 0):
            if i > 100:
                break
            X = data[0].to(device)
            Z_mu, Z_logvar = netE(X)
            emotion_latents[lab].append((Z_mu.view(-1).numpy(), Z_logvar.view(-1).numpy()))

    average_emotion = {}
    for emotion in emotion_latents.keys():
        df_mu = pd.DataFrame([x[0] for x in emotion_latents[emotion]])
        df_logvar = pd.DataFrame([x[1] for x in emotion_latents[emotion]])
        df_mu_average = df_mu.mean().to_numpy()
        df_logvar_average = df_logvar.mean().to_numpy()
        average_emotion[emotion] = (df_mu_average, df_logvar_average)

    with torch.no_grad():
        for emotion in average_emotion.keys():
            input_mu = torch.from_numpy(average_emotion[emotion][0])
            input_logvar = torch.from_numpy(average_emotion[emotion][1])
            reshape = input_mu.reshape([1, cfg['nz'], 1, 1]).float()
            fake = netG(reshape)
            img = vutils.make_grid(fake, padding=5, normalize=True).cpu()
            plt.imshow(img[0], cmap=cm.gray)
            plt.savefig("output_images/emotion_{}.png".format(emotion))