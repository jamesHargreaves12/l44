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
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from VAEGAN_CelebA_emot_change import get_lab_df, get_lab
from emotion_classif_test import get_image_label
from models import Encoder, Generator, Discriminator, initialise, VariationalEncoder
from utils import get_dataset, get_model_and_optimizer, save_images, reparameterize, loss_function_kld, \
    plot_real_vs_fake

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='config file name')
    args = parser.parse_args()

    # Root directory for dataset
    cfg = yaml.load(open(args.config_file))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataset(cfg, shuffle=False)
    original_lab_df, labels = get_lab_df('data/celeba/output_classif.csv')

    netE, optimizerE = get_model_and_optimizer(VariationalEncoder, cfg["enc_path"], cfg)
    netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)

    emotion_latents = defaultdict(list)
    # lab_iter = iter(labs)
    # output_file = open("data/latent_to_emotion.csv", "w+")
    emotion_order = ["Angry", "Happy", "Neutral", "Sad", "Surprise", "Fear"]
    average_emotion = {}
    original = []
    post_recon = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            if not torch.cuda.is_available() and i > 500 and cfg["speedup_emot_change"]:
                break
            X = data[0].to(device)
            filenames = data[2]
            Z_mu, _ = netE(X)
            fake_imgs = netG(Z_mu)
            for img, fname in zip(fake_imgs, filenames):
                orig_lab = get_lab(original_lab_df, fname)
                lab = labels[get_image_label(img)]
                original.append(orig_lab)
                post_recon.append(lab)
    labs = list(set(post_recon))
    print(confusion_matrix(original, post_recon, labs))

