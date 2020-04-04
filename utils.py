from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
from time import time

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
from IPython.display import HTML

from models import Generator, Discriminator, weights_init, initialise


def get_dataset():
    dataroot = "data/FER"
    image_size = 48
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                             shuffle=True, num_workers=2)
    return dataloader


def plot_real_vs_fake(real_imgs, fake_imgs, show=True, save_path=None):
    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_imgs, padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_imgs.cpu(), (1, 2, 0)))
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()



def save_images(imgs, filepath):
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.savefig(filepath)


def get_model_and_optimizer(model_class, model_path, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = 1 if torch.cuda.is_available() else 0
    network = model_class(ngpu, cfg).to(device)
    optimizer = optim.Adam(network.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], 0.999))
    initialise(network, model_path)
    print(network)
    return network, optimizer


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def loss_function_kld(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD
