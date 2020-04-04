# -*- coding: utf-8 -*-
# Code From:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.htmlhttps://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# THis tutorial uses alot of the best practices from the original paper

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
import yaml
from IPython.display import HTML

from models import Generator
from utils import get_dataset, plot_real_vs_fake, get_model_and_optimizer

cfg = yaml.load(open("config.yaml"))

device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
fixed_noise = torch.randn(64, cfg['nz'], 1, 1, device=device)

dataloader = get_dataset()

netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)

with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
fake_imgs = vutils.make_grid(fake, padding=2, normalize=True)

real_batch = next(iter(dataloader))
real_imgs = real_batch[0].to(device)[:64]

plot_real_vs_fake(real_imgs, fake_imgs)
