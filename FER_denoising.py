from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras
from keras.engine.saving import load_model
from keras.layers import Activation, Dense, Input, MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K, Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.optimizers import SGD
import pandas as pd
from sklearn.model_selection import train_test_split


def get_autoencoder(latent_dim, image_size):
    w = image_size
    input = Input(shape=(w, w, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input)  # 48 x 48 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 24 x 24 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 24 x 24 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 12 x 12 x 64
    flat = Flatten()(pool2)  # 9216
    latent = Dense(latent_dim, name='latent_vector')(flat)  # laten_dim
    encoder = Model(input, latent, name='encoder')
    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,))  # laten_dim
    flattened = Dense(w // 4 * w // 4 * 64)(latent_inputs)
    reshaped = Reshape((w // 4, w // 4, 64))(flattened)  # 12 x 12 x 64
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(reshaped)  # 12 x 12 x 64
    up1 = UpSampling2D((2, 2))(conv3)  # 24 x 24 x 64
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 24 x 24 x 32
    up2 = UpSampling2D((2, 2))(conv4)  # 48 x 48 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 48 x 48 x 1
    decoder = Model(latent_inputs, decoded, name='decoder')
    decoder.summary()
    return encoder, decoder

model_file_path = "models/denoise_autoencoder_fer.h5"
np.random.seed(1337)
lab_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
df = pd.read_csv("data/train.csv")

vals = eval(df['pixels'][0])

X = np.array([np.reshape(eval(x), (48, 48)).astype(np.uint8) for x in df['pixels']])
y = np.array(df[lab_cols] / 10)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
latent_dim = 64

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
if os.path.exists(model_file_path):
    autoencoder = load_model(model_file_path)
else:
    encoder, decoder = get_autoencoder(latent_dim, image_size)
    print("Encoder, decoder comiled")
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()

    autoencoder.compile(loss='mse', optimizer='adam')

    autoencoder.fit(x_train_noisy,
                    x_train,
                    validation_data=(x_test_noisy, x_test),
                    epochs=30,
                    batch_size=batch_size)

    autoencoder.save(model_file_path)

# Predict the Autoencoder output from corrupted test images
x_decoded = autoencoder.predict(x_test_noisy)


rows, cols = 1, 5
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()
# Results are as expected very smoothed out due to pixel level loss being used