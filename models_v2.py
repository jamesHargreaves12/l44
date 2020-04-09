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


class Generator(nn.Module):
    def __init__(self, ngpu, cfg):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        nz = cfg["nz"]
        nc = cfg["nc"]
        self.start_w = cfg['image_size'] // 8
        self.expanded_size = 256*self.start_w*self.start_w
        self.expand_state_space = nn.Sequential(
            nn.Linear(nz, self.expanded_size),
            nn.BatchNorm1d(self.expanded_size, momentum=0.9, eps=1e-6),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # state_size = 256 x 8 x 8
            nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9, eps=1e-6),
            nn.ReLU(True),
            # state size. 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9, eps=1e-6),
            nn.ReLU(True),
            # state size. 128 x 32 x 32
            nn.ConvTranspose2d(128, 32, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(32, momentum=0.9, eps=1e-6),
            nn.ReLU(True),
            # state size. 32 x 64 x 64
            nn.ConvTranspose2d(32, nc, 5, 1, 2, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        expanded = self.expand_state_space(input)
        reshaped = expanded.view(-1, 256, self.start_w, self.start_w)
        return self.main(reshaped)


class VariationalEncoder(nn.Module):
    def __init__(self, ngpu, cfg):
        super(VariationalEncoder, self).__init__()
        self.ngpu = ngpu
        self.nc = cfg["nc"]
        self.nz = cfg["nz"]
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(self.nc, 64, 5, 2, 2, bias=False),
            nn.BatchNorm2d(64, momentum=0.9, eps=1e-6),
            nn.ReLU(inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128, momentum=0.9, eps=1e-6),
            nn.ReLU(inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256, momentum=0.9, eps=1e-6),
            nn.ReLU(inplace=True),
            # state size. 256 x 8 x 8
        )
        end_w = cfg['image_size'] // 8
        self.start_latent = 256*end_w*end_w
        self.to_shared_latent = nn.Sequential(
            nn.Linear(self.start_latent, 2048),
            nn.BatchNorm1d(2048, momentum=0.9, eps=1e-6),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(2048, self.nz)
        self.fc_logvar = nn.Linear(2048, self.nz)

    def forward(self, x):
        shared_latent_big = self.main(x)
        shared_latent_flat = shared_latent_big.view(-1, self.start_latent)
        shared_latent_small = self.to_shared_latent(shared_latent_flat)
        mu = self.fc_mu(shared_latent_small)
        logvar = self.fc_logvar(shared_latent_small)
        return mu, logvar


class Discriminator(nn.Module):
    def __init__(self, ngpu, cfg):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        nc = cfg["nc"]

        self.main_to_l = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(nc, 32, 5, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            # state size. 32 x 64 x 64
            nn.Conv2d(32, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128, momentum=0.9, eps=1e-6),
            nn.ReLU(inplace=True),
            # state size. 128 x 32 x 32
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256, momentum=0.9, eps=1e-6),
            nn.ReLU(inplace=True),
            # state size. 256 x 16 x 16
            nn.Conv2d(256, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256, momentum=0.9, eps=1e-6),
            nn.ReLU(inplace=True),
            # state size. 256 x 8 x 8
        )
        end_w = cfg['image_size'] // 8
        self.size_at_l = 256*end_w*end_w
        self.main_after_l = nn.Sequential(
            nn.Linear(self.size_at_l, 512),
            nn.BatchNorm1d(512, momentum=0.9, eps=1e-6),
            nn.ReLU(inplace=True),
            # state size. 512
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        lth = self.main_to_l(input)
        flat = lth.view(-1, self.size_at_l)
        return lth, self.main_after_l(flat)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def initialise(model, path):
    if os.path.exists(path):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path))
        else:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        model.apply(weights_init)


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
