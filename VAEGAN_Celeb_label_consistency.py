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

from emotion_classif_test import get_image_label
from models import Encoder, Generator, Discriminator, initialise, VariationalEncoder
from utils import get_dataset, get_model_and_optimizer, save_images, reparameterize, loss_function_kld, \
    plot_real_vs_fake


def get_lab(labs, id):
    return np.array(labs[labs["image_id"] == id]["lab"])[0]


def get_lab_df(filepath):
    if 'list_attr_celeba.csv' in filepath:
        lab_df = pd.read_csv(filepath)
        lab_df['lab'] = lab_df["Smiling"].map(lambda x: "Smiling" if x == 1 else "Not Smiling")
        labels = ["Smiling", "Not Smiling"]
    elif 'output_classif.csv' in filepath:
        lab_df = pd.read_csv(filepath, header=None)
        lab_df = lab_df.rename(
            columns={0: "image_id", 1: "Angry", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sad", 6: "Surprise",
                     7: "Neutral"})
        # Could add filter based on confidence of prediction here
        expression_order = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        labels = expression_order
        lab_df['lab'] = lab_df[expression_order].idxmax(axis=1)
    elif 'ExpW' in filepath:
        lab_df = pd.read_csv(filepath, header=None)
        expression_order = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                            'unknown']
        lab_df = lab_df.rename(
            columns={0: "image_id", 1: "size", 2: "usage", 3: 'neutral', 4: 'happiness', 5: 'surprise', 6: 'sadness',
                     7: 'anger', 8: 'disgust', 9: 'fear', 10: 'contempt', 11: 'unknown'})
        lab_df['lab'] = lab_df[expression_order].idxmax(axis=1)
        labels = expression_order
    elif 'FER' in filepath:
        labs = open(filepath, 'r').read().split(",")
        labs = [int(x) for x in labs]
        names = ['fer{}.png'.format("0" * (7 - len(str(i))) + str(i)) for i in range(len(labs))]
        lab_df = pd.DataFrame.from_dict({"image_id": names, "lab_int": labs})
        expression_order = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        lab_df['lab'] = np.array(expression_order)[lab_df['lab_int']]
        labels = expression_order
    elif filepath == 'ignore':
        return None, ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    else:
        raise ValueError("Unknown filepath: {}".format(filepath))

    return lab_df[["image_id", "lab"]], labels


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
    print(confusion_matrix(original, post_recon))

                    # output_file.write(",".join([str(x) for x in z]) + "," + lab + "\n")
        # output_file.close()

