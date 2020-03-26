import signal

import torch
import torch.nn as nn
import torch.optim as optim

import IPython
import torchvision

mnist = torchvision.datasets.MNIST(
    root='data/',  # where to put the files
    download=True,  # if files aren't here, download them
    train=True,  # whether to import the test or the train subset
    # PyTorch uses PyTorch tensors internally, not numpy arrays, so convert them.
    transform=torchvision.transforms.ToTensor()
)

# Very many PyTorch building blocks assume that the data comes in batches.
# The DataLoader converts the full mnist list [(img,lbl),...] into batches
#     [(img_batch,lbl_batch), ...]
# where each img_batch is an array with an extra dimension prepended.

mnist_batched = torch.utils.data.DataLoader(mnist, batch_size=5)


def interrupted(_interrupted=[False], _default=[None]):
    if _default[0] is None or signal.getsignal(signal.SIGINT) == _default[0]:
        _interrupted[0] = False

        def handle(signal, frame):
            if _interrupted[0] and _default[0] is not None:
                _default[0](signal, frame)
            print('Interrupt!')
            _interrupted[0] = True

        _default[0] = signal.signal(signal.SIGINT, handle)
    return _interrupted[0]


def enumerate_cycle(g):
    epoch = 0
    while True:
        for i, x in enumerate(g):
            yield (epoch, i), x
        epoch = epoch + 1


class FairAE(nn.Module):
    def __init__(self, latent_dim=4, sensitive_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + sensitive_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1728),
            nn.LeakyReLU(),
            Reshaper((-1, 12, 12, 12)),
            nn.Conv2d(12, 36, 3, 1),
            nn.LeakyReLU(),
            Reshaper((-1, 4, 30, 30)),
            nn.Conv2d(4, 4, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 1, 1, 1),
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
    # Create a new model object, and initialize its parameters
    ae = FairAE(latent_dim=4, sensitive_dim=10)
    adv = Adv(latent_dim=4, sensitive_dim=10)

    # Prepare to iterate through all the training data.
    iter_training_data = enumerate_cycle(mnist_batched)

    ae_optimizer = optim.Adadelta(ae.parameters(), lr=0.1)
    adv_optimizer = optim.Adadelta(adv.parameters(), lr=1.0)

    print("Press Ctrl+C to end training and save parameters")

    while not interrupted():
        (epoch, batch_num), (imgs, lbls) = next(iter_training_data)
        attributes = nn.functional.one_hot(lbls, 10).float()
        z, rx = ae(imgs, attributes)
        l_dec = nn.functional.mse_loss(imgs, rx)
        a = adv(z)
        l_adv = nn.functional.nll_loss(a, lbls)

        ae_optimizer.zero_grad()
        (l_dec - 0.01 * l_adv).backward(retain_graph=True)
        ae_optimizer.step()

        adv_optimizer.zero_grad()
        (-(l_dec - 0.01 * l_adv)).backward()
        adv_optimizer.step()

        if batch_num % 25 == 0:
            IPython.display.clear_output(wait=True)
            print(f'epoch={epoch} batch={batch_num}/{len(mnist_batched)} '
                  f'ae_loss={l_dec.item()} adv_loss={l_adv.item()}')

    # Optionally, save all the parameters
    torch.save(ae.state_dict(), 'fair_ae_4d.pt')
