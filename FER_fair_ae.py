from time import time

import torch
import torch.nn as nn
import torch.optim as optim

import IPython

from dataprep.dataset_fer_plus import FERPlus


class FairAE(nn.Module):
    def __init__(self, latent_dim=4, sensitive_dim=10):
        super().__init__()
        self.encoder = nn.Sequential( #1 x 48 x 48
            nn.Conv2d(1, 32, 3, 1),   #32 x 46 x 46
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1),  #64 x 44 x 44
            nn.MaxPool2d(2),          #64 x 22 x 22
            nn.Flatten(1),            #30975
            nn.Linear(30976, 128),    #128
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + sensitive_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 5808),       # 5808
            nn.LeakyReLU(),
            Reshaper((-1, 12, 22, 22)), # 12 x 22 x 22
            nn.Conv2d(12, 25, 3, 1),    # 25 x 20 x 20
            nn.LeakyReLU(),
            Reshaper((-1, 4, 50, 50)),  #  4 x 50 x 50
            nn.Conv2d(4, 4, 3, 1),      #  4 x 48 x 48
            nn.LeakyReLU(),
            nn.Conv2d(4, 1, 1, 1),      #  1 x 48 x 48
            nn.Sigmoid()
        )

    def forward(self, x, a):
        z = self.encoder(x)
        return z, self.decoder(torch.cat((z, a), 1))


class Reshaper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.reshape(x, self.dim)


class Adv(nn.Module):
    def __init__(self, latent_dim=4, sensitive_dim=10):
        super().__init__()
        self.predicter = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, sensitive_dim),
            nn.LogSoftmax()
        )

    def forward(self, z):
        return self.predicter(z)


if __name__ == '__main__':
    print("Loading data")
    fer = FERPlus('train')
    print("Data loaded")
    fer_batched = torch.utils.data.DataLoader(fer, batch_size=32)
    num_epoch = 30
    print("Loading models")
    ae = FairAE(latent_dim=8, sensitive_dim=10)
    adv = Adv(latent_dim=8, sensitive_dim=10)

    ae_optimizer = optim.Adadelta(ae.parameters(), lr=0.1)
    adv_optimizer = optim.Adadelta(adv.parameters(), lr=1.0)

    for epoch in range(1,num_epoch + 1):
        print("Epoch ", epoch)
        start = 0
        for batch_num, (imgs, lbls) in enumerate(fer_batched):
            attributes = lbls
            z, rx = ae(imgs, attributes)
            l_dec = nn.functional.mse_loss(imgs, rx)
            a = adv(z)
            l_adv = nn.functional.mse_loss(a, lbls)

            ae_optimizer.zero_grad()
            (l_dec - 0.0001 * l_adv).backward(retain_graph=True)
            ae_optimizer.step()

            adv_optimizer.zero_grad()
            (-(l_dec - 0.0001 * l_adv)).backward()
            adv_optimizer.step()

            if batch_num % 100 == 0:
                IPython.display.clear_output(wait=True)
                if start:
                    print(f'epoch={epoch} batch={batch_num}/{len(fer_batched)} time={time()-start}'
                      f'ae_loss={l_dec.item()} adv_loss={l_adv.item()}')
                start = time()

    # Optionally, save all the parameters
    torch.save(ae.state_dict(), 'models/fair_ae_4d.pt')
