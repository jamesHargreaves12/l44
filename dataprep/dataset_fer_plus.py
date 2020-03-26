from __future__ import print_function
import numpy as np
import torch.utils.data as data
import pandas as pd


class FERPlus(data.Dataset):
    def __init__(self, split='train'):
        # rewrite this for speed
        lab_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown',
                    'NF']
        self.df_pixels = pd.read_csv("data/{}_pixels.csv".format(split))
        self.df_labs = pd.read_csv("data/{}_labs.csv".format(split))
        image_size = 48
        self.X = np.reshape(self.df_pixels.to_numpy(), (-1, 1, image_size,image_size)).astype('float32') / 255
        self.y = np.array(self.df_labs / 10).astype('float32')

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]


if __name__ == "__main__":
    x = FERPlus('train')
