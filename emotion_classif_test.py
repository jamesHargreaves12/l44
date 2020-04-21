import os
import sys
import torch
import yaml
from PIL import Image

import numpy as np

from classifer_fer.models import *
from utils import get_dataset

sys.path.append(os.path.join(os.getcwd(), 'classifer_fer'))
import transforms as transforms


transform_test = transforms.Compose([
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

net = VGG('VGG19')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('classifer_fer/FER2013_VGG19/PrivateTest_model.t7', map_location=device)

net.load_state_dict(checkpoint['net'])
# net.cuda()
net.eval()
# cfg = yaml.load(open('config_celeba_grey.yaml', 'r'))

# dataloader = get_dataset(cfg, shuffle=False)
# data = next(iter(dataloader))
# X = data[0].to(torch.device('cpu'))


def get_image_label(img):
    img = img.reshape(48, 48, 1) * 128 + 128
    img = img.numpy().astype(np.uint8)
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    ncrops, c, h, w = np.shape(inputs)
    # print(ncrops, c, h, w)
    inputs = inputs.view(-1, c, h, w)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    inputs = Variable(inputs)
    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    _, predicted = torch.max(outputs_avg.data, 0)
    return int(predicted.view(-1)[0])
