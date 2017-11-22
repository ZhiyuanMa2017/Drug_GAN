#tensorboard --logdir=D:\GitHub\tf_demo\graphs


import numpy as np
import tensorflow as tf
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from load_cifar10 import load_CIFAR10

cifar10_dir = 'data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X_train_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3)
X_train = minmax.fit_transform(X_train_rows)
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
fig, axes = plt.subplots(nrows=4, ncols=20, sharex=True, sharey=True, figsize=(80, 16))
imgs = X_train[y_train==0][:80]
for image, row in zip([imgs[:20], imgs[20:40], imgs[40:60], imgs[60:80]], axes):
    for img, ax in zip(image, row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.savefig('original000.jpg')
