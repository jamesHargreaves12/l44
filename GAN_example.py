# -*- coding: utf-8 -*-
# Code From:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
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

from models import Generator, Discriminator, weights_init, initialise
from utils import get_dataset, get_model_and_optimizer, plot_real_vs_fake, save_images

if __name__ == "__main__":

    # Root directory for dataset
    cfg = yaml.load(open("config.yaml"))

    device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
    fixed_noise = torch.randn(64, cfg['nz'], 1, 1, device=device)

    dataloader = get_dataset()

    netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)
    netD, optimizerD = get_model_and_optimizer(Discriminator, cfg["dis_path"], cfg)

    criterion = nn.BCELoss()

    img_list = []

    print("Starting Training Loop...")
    for epoch in range(cfg["num_epoch"]):
        start = time()
        print("Epoch:", epoch)
        for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # real batch
            netD.zero_grad()
            # Format batch
            label = torch.full((b_size,), cfg["real_label"], device=device)
            _, output = netD(real_cpu)
            output = output.view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            D_x = output.mean().item()

            # all-fake batch
            noise = torch.randn(b_size, cfg["nz"], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(cfg["fake_label"])
            _, output = netD(fake.detach())
            output = output.view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()
            # D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(cfg["real_label"])  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            _, output = netD(fake)
            output = output.view(-1)
            errG = criterion(output, label)
            errG.backward()
            # D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % cfg["image_rate"] == 0 or ((epoch == cfg["num_epoch"] - 1) and (i == len(dataloader) - 1)):
                print(
                    "Time: {:.1f} {:d}/{:d}\tLoss_D: {:.4f}\tLoss_G: {:.4f}".format(time() - start, i, len(dataloader),
                                                                                    errD.item(), errG.item()))
                start = time()

                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                fake_imgs = vutils.make_grid(fake, padding=2, normalize=True)
                save_images(fake_imgs, "output_images/GAN_out_{}_{}.png".format(epoch,i))

    torch.save(netG.state_dict(), cfg["gen_path"])
    torch.save(netD.state_dict(), cfg["dis_path"])

    real_batch = next(iter(dataloader))
    real_imgs = real_batch[0].to(device)[:64]
    if not torch.cuda.is_available():
        plot_real_vs_fake(real_imgs, img_list[-1])
