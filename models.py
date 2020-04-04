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
        ngf = cfg["ngf"]
        nc = cfg["nc"]
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 3 x 3
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 6 x 6
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 12 x 12
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 24 x 24
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 48 x 48
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, ngpu, cfg):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        nc = cfg["nc"]
        ndf = cfg["ndf"]
        nz = cfg["nz"]
        self.main = nn.Sequential(
            # input is (nc) x 48 x 48
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 24 x 24
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, nz, 3, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class VariationalEncoder(nn.Module):
    def __init__(self, ngpu, cfg):
        super(VariationalEncoder, self).__init__()
        self.ngpu = ngpu
        self.nc = cfg["nc"]
        self.ndf = cfg["ndf"]
        self.nz = cfg["nz"]
        self.main = nn.Sequential(
            # input is (nc) x 48 x 48
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 24 x 24
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(self.ndf * 4, 2*self.nz, 3, 4, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        print(self.main)
        self.fc_mu = nn.Linear(2*self.nz, self.nz)
        self.fc_logvar = nn.Linear(2*self.nz, self.nz)

    def forward(self, x):
        shared_latent = self.main(x)
        shared_latent = shared_latent.view(-1, 2*self.nz)
        mu = self.fc_mu(shared_latent)
        logvar = self.fc_logvar(shared_latent)
        return mu, logvar


class Discriminator(nn.Module):
    def __init__(self, ngpu, cfg):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        nc = cfg["nc"]
        ndf = cfg["ndf"]
        self.main_to_l = nn.Sequential(
            # input is (nc) x 48 x 48
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 24 x 24
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main_after_l = nn.Sequential(
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        lth = self.main_to_l(input)

        return lth, self.main_after_l(lth)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def initialise(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        model.apply(weights_init)


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
