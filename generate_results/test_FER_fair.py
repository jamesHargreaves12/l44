import torch
import torchvision
import matplotlib.pyplot as plt

from old.FER_fair_ae import FairAE
from dataprep.dataset_fer_plus import FERPlus

ae = FairAE(latent_dim=128, sensitive_dim=10)
print("Loading Model")
ae.load_state_dict(torch.load('fair_ae_4d.pt'))
print("Loading Data")
fer = FERPlus('train')
fer_batched = torch.utils.data.DataLoader(fer, batch_size=5)

print("Running Generator")
for imgs,atts in fer_batched:
    z,rx = ae(imgs,atts)
    img_recon = torch.cat([imgs, rx.detach()], 0)
    x = torchvision.utils.make_grid(img_recon, nrow=5, pad_value=.5)

    plt.imshow(x.numpy().transpose((1, 2, 0)))
    plt.show()
    break