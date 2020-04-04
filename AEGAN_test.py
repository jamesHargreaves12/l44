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
from IPython.display import HTML

from models import Encoder, Generator, Discriminator, initialise

if __name__ == "__main__":

    # Root directory for dataset
    dataroot = "data/FER"
    gen_path = "models/Gen_aegan_2.trc"
    dis_path = "models/Dis_aegan_2.trc"
    enc_path = "models/Enc_aegan_2.trc"

    lambda_val = 0.1
    image_size = 48
    num_epochs = 100

    nc = 1  # Number of channels
    nz = 1  # Size of z latent vector (i.e. size of generator input)
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    lr = 0.0002
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    ngpu = 1 if torch.cuda.is_available() else 0 # Number of GPUs available. Use 0 for CPU mode.

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                             shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netE = Encoder(ngpu, nc, ndf, nz).to(device)
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netD = Discriminator(ngpu, ndf, nc).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netE = nn.DataParallel(netE, list(range(ngpu)))
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    calc_BCE_loss = nn.BCELoss()
    calc_MSE_loss = nn.MSELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerE = optim.Adam(netE.parameters(), lr=lr, betas=(beta1, 0.999))

    initialise(netE, enc_path)
    initialise(netG, gen_path)
    initialise(netD, dis_path)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        start = time()
        print("Epoch:", epoch)
        for i, data in enumerate(dataloader, 0):
            # Encode data
            X = data[0].to(device)
            b_size = X.size(0)


            # Update the Discriminator Network
            netD.zero_grad()
            # Real
            label = torch.full((b_size,), real_label, device=device)
            _, Dis_X = netD(X.detach())
            errD_real = calc_BCE_loss(Dis_X.view(-1), label)
            errD_real.backward()
            D_x = Dis_X.view(-1).mean().item()

            # Fake / Generated
            Z = netE(X.detach())
            X_tilde = netG(Z.detach())
            label.fill_(fake_label)
            _, Dis_X_tilde = netD(X_tilde.detach())
            errD_fake = calc_BCE_loss(Dis_X_tilde.view(-1), label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator Network
            netG.zero_grad()
            Z = netE(X.detach())
            # possible
            Dis_l_X, _ = netD(X)
            X_tilde = netG(Z)
            label.fill_(real_label)
            # possible
            Dis_l_X_tilde, Dis_X_tilde = netD(X_tilde)
            errG_style = calc_BCE_loss(Dis_X_tilde.view(-1), label)
            err_content = calc_MSE_loss(Dis_l_X_tilde.view(-1), Dis_l_X.view(-1))
            errG = errG_style + lambda_val*err_content
            errG.backward()
            # D_G_z2 = output.mean().item()
            optimizerG.step()

            netE.zero_grad()
            Z = netE(X)
            X_tilde = netG(Z)
            Dis_l_X_tilde, Dis_X_tilde = netD(X_tilde)
            Dis_l_X, _ = netD(X)
            err_content = calc_MSE_loss(Dis_l_X_tilde.view(-1), Dis_l_X.view(-1))
            err_content.backward()
            optimizerE.step()

            # Output training stats
            if i % 50 == 0:
                print("Time: {:.1f} {:d}/{:d}\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tLoss_E: {:.4f}".format(time() - start, i, len(dataloader),
                                                                                      errD.item(), errG.item(),err_content.item()))
                start = time()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    torch.save(netG.state_dict(), gen_path)
    torch.save(netD.state_dict(), dis_path)
    torch.save(netE.state_dict(), enc_path)

    # # Analysis
    # plt.figure(figsize=(10, 5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses, label="G")
    # plt.plot(D_losses, label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
    #
    # # %%capture
    # fig = plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    #
    # HTML(ani.to_jshtml())
    #
    # ######################################################################
    # # **Real Images vs. Fake Images**
    # #
    # # Finally, lets take a look at some real images and fake images side by
    # # side.
    # #
    #
    # # Grab a batch of real images from the dataloader
    # real_batch = next(iter(dataloader))
    #
    # # Plot the real images
    # plt.figure(figsize=(15, 15))
    # plt.subplot(1, 2, 1)
    # plt.axis("off")
    # plt.title("Real Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    #
    # # Plot the fake images from the last epoch
    # plt.subplot(1, 2, 2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    # plt.show()
