import torch
import yaml
from torch import optim, nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import seaborn as sn

from models import LatentEmotionPredict, initialise
import pandas as pd
import numpy as np


class LatentDataset(Dataset):
    def __init__(self):
        data_path = "data/latent_to_emotion.csv"

        data = pd.read_csv(data_path, header=None)
        labs = data.pop(100)

        self.labels = labs
        self.feats = data

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        X = torch.from_numpy(np.array(self.feats.iloc[index]))

        one_hot = np.zeros(7)
        one_hot[self.labels[index]] = 1
        y = torch.from_numpy(one_hot)
        return X, y


cfg = yaml.load(open("config_aegan.yaml", "r"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LatentEmotionPredict(cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], 0.999))
calc_BCE = nn.BCELoss()

training_set = LatentDataset()
training_generator = DataLoader(training_set, batch_size=64)


for epoch in range(cfg["num_epoch"]):
    print("Epoch:", epoch)
    for i, (d, l) in enumerate(training_generator, 0):
        X = d.to(device).float()
        model.zero_grad()
        out = model(X)
        err = calc_BCE(out, l.float())
        err.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch {} {} loss:{} ".format(epoch, i, err.item()))

pred = []
true = []
with torch.no_grad():
    for d, l in training_generator:
        X = d.to(device).float()
        y_pred = model(X)
        pred.extend([int(np.argmax(y)) for y in y_pred])
        true.extend([int(np.argmax(y)) for y in y_pred])
x=1
print(pred)
#Best predictor just predicts the most common class every time