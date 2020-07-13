import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time
from pandas import Series

np.random.seed(2017)
episodes = 30000
data = pd.read_csv('stahl.csv')
Y = data.SMILES
Y.head()
X = data.ix[:, 1:7]
X = X.values
X = X.astype('int')
type(X)


Ylist=list(Y)
Ynewlist=list()
for i in range(len(Ylist)):
    Ynewlist.append(Ylist[i][0:28])
Y = Series(Ynewlist)

maxY = Y.str.len().max() + 7
#y = Y.str.ljust(maxY, fillchar='|')
y=Y
ts = y.str.len().max()
print ("ts={0}".format(ts))
# CharToIndex and IndexToChar functions
chars = sorted(list(set("".join(y.values.flatten()))))
print('total chars:', len(chars))

char_idx = dict((c, i) for i, c in enumerate(chars))
idx_char = dict((i, c) for i, c in enumerate(chars))

def dimY(Y, ts, char_idx, chars):
    temp = np.zeros((len(Y), ts, len(chars)))
    for i, c in enumerate(Y):
        for j, s in enumerate(c):
            # print i, j, s
            temp[i, j, char_idx[s]] = 1
    return np.array(temp)
y_dash = dimY(y, ts, char_idx, chars)
temp = np.zeros((28,28-19))
Y_dash=np.zeros((len(Y),28,28))
for i in range(len(y_dash)):
    Y_dash[i]=np.concatenate((y_dash[0],temp),axis=1)
x_dash = X
print("X={0} Y={1}".format(x_dash.shape, Y_dash.shape))

def prediction(preds):
    y_pred = []
    for i, c in enumerate(preds):
        y_pred.append([])
        for j in c:
            y_pred[i].append(np.argmax(j))
    return np.array(y_pred)
# sequence to text conversion

def seq_txt(y_pred, idx_char):
    newY = []
    for i, c in enumerate(y_pred):
        newY.append([])
        for j in c:
            newY[i].append(idx_char[j])

    return np.array(newY)

# joined smiles output

def smiles_output(s):
    smiles = np.array([])
    for i in s:
        j = ''.join(str(k) for k in i)
        smiles = np.append(smiles, j)
    return smiles


x_pred = [[0, 0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1]]

# preds = Ghash.predict(x_pred)
# y_pred = prediction(preds)
# y_pred = seq_txt(y_pred, idx_char)
# s = smiles_output(y_pred)
# print(s)

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)


def sample_noise(batch_size, dim):
    # return tf.random_uniform([batch_size, dim], minval=-1, maxval=1)
    return tf.random_normal([batch_size, dim])

def discriminator(x):
    with tf.variable_scope("discriminator"):
        x = tf.reshape(x, [-1, 784])
        layer1 = tf.layers.dense(x, units=20,use_bias=True)
        layer2 = tf.layers.batch_normalization(layer1, training=is_training)
        layer3 = leaky_relu(layer2)
        layer4 = tf.layers.dense(layer3, units=20,use_bias=True)
        layer5 = tf.layers.batch_normalization(layer4, training=is_training)
        layer6 = leaky_relu(layer5)

        logits = tf.layers.dense(layer6, units=1)
        return logits


def generator(z):
    with tf.variable_scope("generator"):
        layer1 = tf.layers.dense(z, units=1024)
        layer2 = tf.layers.batch_normalization(layer1, training=is_training)
        layer3 = tf.nn.relu(layer2)
        layer4 = tf.layers.dense(layer3, units=7 * 7 * 128)
        layer5 = tf.layers.batch_normalization(layer4, training=is_training)
        layer6 = tf.nn.relu(layer5)
        imgresize = tf.reshape(layer6, [-1, 7, 7, 128])
        layer7 = tf.layers.conv2d_transpose(imgresize, filters=64, kernel_size=4, strides=2, padding='SAME')
        layer8 = tf.layers.batch_normalization(layer7, training=is_training)
        layer9 = tf.nn.relu(layer8)
        img = tf.layers.conv2d_transpose(layer9, filters=1, kernel_size=4, strides=2, padding='SAME',
                                         activation=tf.tanh)
        return img

tf.reset_default_graph()


def get_solvers(learning_rate=1e-3, beta1=0.5):
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    pass
    return D_solver, G_solver

batch_size = 32
# our noise dimension
noise_dim = 10

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 28, 28])
z = sample_noise(batch_size, noise_dim)
is_training = tf.placeholder(tf.bool)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(preprocess_img(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)


def gan_loss(logits_real, logits_fake):
    ones = tf.ones_like(logits_fake)
    # nines = tf.fill(dims=ones.shape, value=0.9, name=None)
    zeros = tf.zeros_like(logits_fake)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=ones))
    D_loss = 0
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=ones))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=zeros))
    return D_loss, G_loss

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')

D_solver,G_solver = get_solvers()
D_loss, G_loss = gan_loss(logits_real, logits_fake)
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'generator')
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)
summ = tf.summary.merge_all()

show_every=250
print_every=50
num_epoch=100

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    for epoch in range(num_epoch):
        total_batch = int(len(Y_dash) / batch_size)
        for i in range(total_batch):
            batch_x = Y_dash[i:i + batch_size]
            _,  __= sess.run([D_train_step,G_train_step], feed_dict={x: batch_x,is_training: True})
            for j in range(10):
                _ = sess.run([G_train_step], feed_dict={is_training: True})
            D_loss_curr,  G_loss_curr, summm = sess.run([D_loss, G_loss, summ], feed_dict={x: batch_x,is_training: True})
            # _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={is_training: True})
            writer.add_summary(summm)

        print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))

