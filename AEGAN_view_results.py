from __future__ import print_function
import os
import random
from time import time

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
from IPython.display import HTML

from models import Generator, Discriminator, VariationalEncoder
from utils import get_dataset, get_model_and_optimizer, plot_real_vs_fake

cfg = yaml.load(open("config_aegan.yaml"))

device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

dataloader = get_dataset(shuffle=False)

netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)
netE, optimizerE = get_model_and_optimizer(VariationalEncoder, cfg["enc_path"], cfg)

test_batch = next(iter(dataloader))
test_imgs = test_batch[0].to(device)[:64]


with torch.no_grad():
    Z_mu, _ = netE(test_imgs)
    fake = netG(Z_mu.reshape(-1, cfg['nz'], 1, 1))

fake_imgs = vutils.make_grid(fake, padding=2, normalize=True)[:64]
plot_real_vs_fake(test_imgs, fake_imgs, show=False,
                  save_path="output_images/VAEGAN_out.png")
