import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from load_cifar10 import load_CIFAR10
cifar10_dir = 'data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
X = np.concatenate((X_train,X_test))
y = np.concatenate((y_train,y_test))


fig, axes = plt.subplots(nrows=4, ncols=20, sharex=True, sharey=True, figsize=(80, 16))
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X_rows = X.reshape(X.shape[0], 32 * 32 * 3)
X = minmax.fit_transform(X_rows)
X = X.reshape(X.shape[0], 32, 32, 3)
imgs = X[y==4][:80]
for image, row in zip([imgs[:20], imgs[20:40], imgs[40:60], imgs[60:80]], axes):
    for img, ax in zip(image, row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.savefig('cifar10-4.jpg')