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

from models import Encoder, Generator, Discriminator, initialise, VariationalEncoder
from utils import get_dataset, get_model_and_optimizer, save_images, reparameterize, loss_function_kld

if __name__ == "__main__":

    # Root directory for dataset
    lambda_val = 0.1
    cfg = yaml.load(open("config_aegan.yaml"))

    device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
    fixed_noise = torch.randn(64, cfg['nz'], 1, 1, device=device)
    print("FIXED NOISE", fixed_noise.shape)
    dataloader = get_dataset()

    netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)
    netD, optimizerD = get_model_and_optimizer(Discriminator, cfg["dis_path"], cfg)
    netE, optimizerE = get_model_and_optimizer(VariationalEncoder, cfg["enc_path"], cfg)

    criterion = nn.BCELoss()

    calc_BCE_loss = nn.BCELoss()
    calc_MSE_loss = nn.MSELoss()

    img_list = []

    start = time()
    for epoch in range(cfg["num_epoch"]):
        print("Epoch:", epoch)
        for i, data in enumerate(dataloader, 0):
            X = data[0].to(device)
            b_size = X.size(0)

            # Update the Discriminator Network
            netD.zero_grad()
            Z_mu, Z_logvar = netE(X)
            Z_mu = Z_mu.detach()
            Z_logvar = Z_logvar.detach()
            # Real
            label = torch.full((b_size,), cfg["real_label"], device=device)
            _, Dis_X = netD(X)
            Dis_x = Dis_X.view(-1)
            errD_real = calc_BCE_loss(Dis_x, label)
            errD_real.backward()

            # Fake
            X_tilde = netG(Z_mu.reshape(-1,cfg['nz'],1,1))
            label.fill_(cfg["fake_label"])
            _, Dis_X_tilde = netD(X_tilde)
            Dis_X_tilde = Dis_X_tilde.view(-1)
            errD_fake = calc_BCE_loss(Dis_X_tilde, label)
            errD_fake.backward()

            # Sampled
            Zp = reparameterize(Z_mu, Z_logvar)
            Xp = netG(Zp.reshape(-1, cfg['nz'], 1, 1)).detach()
            label.fill_(cfg["fake_label"])
            _, Dis_Xp = netD(Xp)
            Dis_Xp = Dis_Xp.view(-1)
            errD_resamp = calc_BCE_loss(Dis_Xp, label)
            errD_resamp.backward()
            #
            errD = errD_real + errD_fake + errD_resamp
            optimizerD.step()

            # Update Generator Network
            netG.zero_grad()

            Dis_l_X, _ = netD(X)
            Z_mu, Z_logvar = netE(X)
            Z_mu = Z_mu.detach()
            Z_logvar = Z_logvar.detach()
            Zp = reparameterize(Z_mu, Z_logvar)

            X_tilde = netG(Z_mu.reshape(-1, cfg['nz'], 1, 1))
            Dis_l_X_tilde, Dis_X_tilde = netD(X_tilde)

            Xp = netG(Zp.reshape(-1, cfg['nz'], 1, 1))
            label.fill_(cfg["fake_label"])
            _, Dis_Xp = netD(Xp)
            Dis_Xp = Dis_Xp.view(-1)

            err_content = calc_MSE_loss(Dis_l_X_tilde.view(-1), Dis_l_X.view(-1))
            label.fill_(cfg["real_label"])
            errG_style_1 = calc_BCE_loss(Dis_X_tilde.view(-1), label)
            errG_style_2 = calc_BCE_loss(Dis_Xp.view(-1), label)

            errG = errG_style_1 + errG_style_2 + lambda_val * err_content
            errG.backward()
            optimizerG.step()
            #
            # # Variation encoder
            netE.zero_grad()
            Dis_l_X, _ = netD(X)

            Z_mu, Z_logvar = netE(X)
            X_tilde = netG(Z_mu.reshape(-1, cfg['nz'], 1, 1))
            Dis_l_X_tilde, _ = netD(X_tilde)
            err_content = calc_MSE_loss(Dis_l_X_tilde.view(-1), Dis_l_X.view(-1))
            err_distribution = loss_function_kld(Z_mu, Z_logvar)
            errE = err_content + err_distribution
            errE.backward()
            optimizerE.step()

            # # Output training stats
            if i % cfg["image_rate"] == 0 or ((epoch == cfg["num_epoch"] - 1) and (i == len(dataloader) - 1)):
                out_format = "Time: {:.1f} {:d}/{:d}\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tLoss_E: {:.4f}"
                print(
                    out_format.format(time() - start, i, len(dataloader), errD.item(), errG.item(), err_content.item()))
                start = time()

                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                fake_imgs = vutils.make_grid(fake, padding=2, normalize=True)
                save_images(fake_imgs, "output_images/VAEGAN_out_{}_{}.png".format(epoch, i))

    torch.save(netG.state_dict(), cfg["gen_path"])
    torch.save(netD.state_dict(), cfg["dis_path"])
    torch.save(netE.state_dict(), cfg["enc_path"])
