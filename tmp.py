import pandas as pd
import numpy as np
from keras.datasets import mnist

lab_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
df = pd.read_csv("data/train.csv")

vals = eval(df['pixels'][0])


(x_train, _), (x_test, _) = mnist.load_data()
X = np.array([np.reshape(eval(x), (48, 48)).astype(np.uint8) for x in df['pixels']])
y = np.array(df[lab_cols] / 10)
x=1