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

from models import Generator, Discriminator, VariationalEncoder, DiscriminatorWGAN
from utils import get_dataset, get_model_and_optimizer, save_images, reparameterize, loss_function_kld, \
    plot_real_vs_fake

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='config file name')
    args = parser.parse_args()

    # Root directory for dataset
    cfg = yaml.load(open(args.config_file))
    dataloader = get_dataset(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

    netG, optimizerG = get_model_and_optimizer(Generator, cfg["gen_path"], cfg)
    netD, optimizerD = get_model_and_optimizer(DiscriminatorWGAN, cfg["dis_path"], cfg)
    netE, optimizerE = get_model_and_optimizer(VariationalEncoder, cfg["enc_path"], cfg)

    calc_BCE_loss = nn.BCELoss()
    calc_MSE_loss = nn.MSELoss()

    test_batch = next(iter(dataloader))
    test_imgs = test_batch[0].to(device)[:64]
    iteration = 0
    diter = 5
    start = time()
    one = torch.FloatTensor([1]*64)
    minus_one = one * -1
    label = torch.full((64,), cfg["real_label"], device=device)

    data_iter = iter(dataloader)
    for epoch in range(cfg["num_epoch"]):
        print("Epoch:", epoch)
        i = 0
        while i < len(dataloader):
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            j = 0
            while j < diter and i < len(dataloader):
                j += 1
                i += 1
                data = next(data_iter)
                for p in netD.parameters():
                    p.data.clamp_(-0.01, 0.01)

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
                errD_real = Dis_X.view(-1)
                errD_real.backward(one)

                # Fake
                X_tilde = netG(Z_mu)
                label.fill_(cfg["fake_label"]).to(device)
                _, Dis_X_tilde = netD(X_tilde)
                errD_fake = Dis_X_tilde.view(-1)
                errD_fake.backward(minus_one)

                # Sampled
                Zp = reparameterize(Z_mu, Z_logvar)
                Xp = netG(Zp).detach()
                label.fill_(cfg["fake_label"]).to(device)
                _, Dis_Xp = netD(Xp)
                # Dis_Xp = Dis_Xp.view(-1)
                # errD_resamp = calc_BCE_loss(Dis_Xp, label)
                errD_resamp = Dis_Xp.view(-1)
                errD_resamp.backward(minus_one)
                #
                errD = errD_real - errD_fake - errD_resamp
                optimizerD.step()

            # for i, /data in enumerate(dataloader, 0):

            # Update Generator Network
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            Dis_l_X, _ = netD(X)
            Z_mu, Z_logvar = netE(X)
            Z_mu = Z_mu.detach()
            Z_logvar = Z_logvar.detach()
            Zp = reparameterize(Z_mu, Z_logvar)

            X_tilde = netG(Z_mu)
            Dis_l_X_tilde, Dis_X_tilde = netD(X_tilde)

            Xp = netG(Zp)
            label.fill_(cfg["fake_label"]).to(device)
            _, Dis_Xp = netD(Xp)
            Dis_Xp = Dis_Xp.view(-1)

            err_content = calc_MSE_loss(Dis_l_X_tilde.view(-1), Dis_l_X.view(-1))
            label.fill_(cfg["real_label"]).to(device)
            errG_style_1 = Dis_X_tilde.view(-1)
            errG_style_2 = Dis_Xp.view(-1)
            # errG_style_1 = calc_BCE_loss(Dis_X_tilde.view(-1), label)
            # errG_style_2 = calc_BCE_loss(Dis_Xp.view(-1), label)
            errG_style_1.backward(one, retain_graph=True)
            errG_style_2.backward(one)
            err_content = cfg["lambda_val"] * err_content
            err_content.backward()
            errG = errG_style_1 + errG_style_2 + err_content
            optimizerG.step()
            #
            # # Variation encoder
            netE.zero_grad()
            Dis_l_X, _ = netD(X)

            Z_mu, Z_logvar = netE(X)
            X_tilde = netG(Z_mu)
            Dis_l_X_tilde, _ = netD(X_tilde)
            err_content = calc_MSE_loss(Dis_l_X_tilde.view(-1), Dis_l_X.view(-1))
            err_distribution = loss_function_kld(Z_mu, Z_logvar)
            errE = err_content + err_distribution
            errE.backward()
            optimizerE.step()

            # # Output training stats
            if iteration % cfg["image_rate"] < diter or ((epoch == cfg["num_epoch"] - 1) and (i == len(dataloader) - 1)):
                out_format = "Time: {:.1f} {:d}/{:d}\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tLoss_E: {:.4f}"
                print(
                    out_format.format(time() - start, i, len(dataloader), errD.data.mean(), errG.data.mean(), errE.item()))
                start = time()

                with torch.no_grad():
                    Z_mu, _ = netE(test_imgs)
                    fake = netG(Z_mu)

                fake_imgs = vutils.make_grid(fake, padding=2, normalize=True)[:64]
                plot_real_vs_fake(test_imgs, fake_imgs, show=False,
                                  save_path="output_images/WGAN_out_{}_{}.png"
                                  .format(epoch, i))
            iteration += 1
        print("Saving")
        torch.save(netG.state_dict(), cfg["gen_path"])
        torch.save(netD.state_dict(), cfg["dis_path"])
        torch.save(netE.state_dict(), cfg["enc_path"])
