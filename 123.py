#tensorboard --logdir=D:\GitHub\tf_demo\graphs


# import numpy as np
# import tensorflow as tf
# import pickle
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from load_cifar10 import load_CIFAR10

# cifar10_dir = 'data/cifar-10-batches-py'
# X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# fig, axes = plt.subplots(nrows=4, ncols=20, sharex=True, sharey=True, figsize=(80, 16))
# imgs = X_train[y_train==0][:80]
# for image, row in zip([imgs[:20], imgs[20:40], imgs[40:60], imgs[60:80]], axes):
#     for img, ax in zip(image, row):
#         ax.imshow(img.astype('uint8'))
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0.1)
# plt.savefig('original000.jpg')


# from sklearn.preprocessing import MinMaxScaler
# minmax = MinMaxScaler()
# X_train_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3)
# X_train = minmax.fit_transform(X_train_rows)
# X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
# fig, axes = plt.subplots(nrows=4, ncols=20, sharex=True, sharey=True, figsize=(80, 16))
# imgs = X_train[y_train==0][:80]
# for image, row in zip([imgs[:20], imgs[20:40], imgs[40:60], imgs[60:80]], axes):
#     for img, ax in zip(image, row):
#         ax.imshow(img)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0.1)
# plt.savefig('after000.jpg')

# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.savefig('cs231nplot.jpg')

# from sklearn.preprocessing import MinMaxScaler
# minmax = MinMaxScaler()
# images = np.concatenate((X_train[y_train==2],X_test[y_test==2]))
# images_rows = images.reshape(images.shape[0], 32 * 32 * 3)
# images = minmax.fit_transform(images_rows)
# images = images.reshape(images.shape[0], 32, 32, 3)


import os
from PIL import Image

def image_resize(img, size=(1500, 1100)):
    if img.mode not in ('L', 'RGB'):
        img = img.convert('RGB')
        img = img.resize(size)

    return img


def image_merge(images, output_dir='output', output_name='merge.jpg',restriction_max_width=None, restriction_max_height=None):

    max_width = 0
    total_height = 0
    # 计算合成后图片的宽度（以最宽的为准）和高度
    for img_path in images:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
            if width > max_width:
                max_width = width
            total_height += height

            # 产生一张空白图
    new_img = Image.new('RGB', (max_width, total_height), 255)
    # 合并
    x = y = 0
    for img_path in images:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
            new_img.paste(img, (x, y))
            y += height

    if restriction_max_width and max_width >= restriction_max_width:
        # 如果宽带超过限制
        # 等比例缩小
        ratio = restriction_max_height / float(max_width)
        max_width = restriction_max_width
        total_height = int(total_height * ratio)
        new_img = image_resize(new_img, size=(max_width, total_height))

    if restriction_max_height and total_height >= restriction_max_height:
        # 如果高度超过限制
        # 等比例缩小
        ratio = restriction_max_height / float(total_height)
        max_width = int(max_width * ratio)
        total_height = restriction_max_height
        new_img = image_resize(new_img, size=(max_width, total_height))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = '%s/%s' % (output_dir, output_name)
    new_img.save(save_path)
    return save_path

imageslist = []
for e in range(10000):
    if e % 500 == 0:
        imageslist.append('111final-ep' + str(e) + '.jpg')

if __name__ == '__main__':
    image_merge(images=imageslist)