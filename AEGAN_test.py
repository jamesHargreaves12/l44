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
    print(model)


class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
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


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
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


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
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


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "data/FER"
gen_path = "Models/Gen_aegan.trc"
dis_path = "Models/Dis_aegan.trc"
enc_path = "Models/Enc_aegan.trc"

lambda_val = 0.1
image_size = 48
num_epochs = 100

nc = 1  # Number of channels
nz = 100  # Size of z latent vector (i.e. size of generator input)
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

# Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.show()

netE = Encoder(ngpu).to(device)
netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)

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
torch.save(netD.state_dict(), enc_path)

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
