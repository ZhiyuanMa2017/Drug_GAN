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

data = pd.read_excel('13.xls')
Y = data.SMILES

Ylist=list(Y)
Ynewlist=list()
for i in range(len(Ylist)):
    Ynewlist.append(Ylist[i][0:28])
Y = Series(Ynewlist)

# maxY = Y.str.len().max()
# y = Y.str.ljust(maxY, fillchar='|')
y=Y
ts = y.str.len().max()
print ("ts={0}".format(ts))
# CharToIndex and IndexToChar functions
chars = sorted(list(set("".join(y.values.flatten()))))
print('total chars:', len(chars))

char_idx = dict((c, i) for i, c in enumerate(chars))
idx_char = dict((i, c) for i, c in enumerate(chars))

def dimY(Y, ts, char_idx, chars):
    temp = np.zeros((len(Y), ts, len(chars)), dtype=np.bool)
    for i, c in enumerate(Y):
        for j, s in enumerate(c):
            # print i, j, s
            temp[i, j, char_idx[s]] = 1
    return np.array(temp)
Y_dash = dimY(y, ts, char_idx, chars)
# temp = np.zeros((28,28-19), dtype=np.bool)
# Y_dash=np.zeros((len(Y),28,28), dtype=np.bool)
# for i in range(len(y_dash)):
#     Y_dash[i]=np.concatenate((y_dash[0],temp),axis=1)

print("Y={0}".format(Y_dash.shape))

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





batch_size = 128
lr_generator = 0.001
lr_discriminator = 0.001

image_dim = 784
noise_dim = 10

noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
is_training = tf.placeholder(tf.bool)

def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        layer1 = tf.layers.dense(x, units=1024)
        layer2 = tf.layers.batch_normalization(layer1, training=is_training)
        layer3 = tf.nn.relu(layer2)
        layer4 = tf.layers.dense(layer3, units=7 * 7 * 128)
        layer5 = tf.layers.batch_normalization(layer4, training=is_training)
        layer6 = tf.nn.relu(layer5)
        imgresize = tf.reshape(layer6, [-1, 7, 7, 128])
        layer7 = tf.layers.conv2d_transpose(imgresize, filters=64, kernel_size=4, strides=2, padding='SAME')
        layer8 = tf.layers.batch_normalization(layer7, training=is_training)
        layer9 = tf.nn.relu(layer8)
        img = tf.layers.conv2d_transpose(layer9, filters=1, kernel_size=4, strides=2, padding='SAME')
        img = tf.tanh(img)
        return img


def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.reshape(x, shape=[-1, 7 * 7 * 128])
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.layers.dense(x, 2)
    return x


gen_sample = generator(noise_input)

disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)

stacked_gan = discriminator(gen_sample, reuse=True)

disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))
disc_loss = disc_loss_real + disc_loss_fake
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=tf.ones([batch_size], dtype=tf.int32)))

optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator, beta1=0.5, beta2=0.999)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
with tf.control_dependencies(gen_update_ops):
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)

disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
with tf.control_dependencies(disc_update_ops):
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)


tf.summary.scalar('D_loss', disc_loss)
tf.summary.scalar('G_loss', gen_loss)
gdsummary = tf.summary.merge_all()

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
writer = tf.summary.FileWriter('./graphs', sess.graph)

epochs=10000

smilesganlist=list()
for e in range(epochs):
    for batch_i in range(4445 // batch_size - 1):
        batch_x = Y_dash[batch_i * batch_size: (batch_i + 1) * batch_size]
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
        batch_x = batch_x * 2. - 1.

        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, dl = sess.run([train_disc, disc_loss],
                         feed_dict={real_image_input: batch_x, noise_input: z, is_training: True})

        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, gl = sess.run([train_gen, gen_loss], feed_dict={noise_input: z, is_training: True})

        gdsum = sess.run(gdsummary, feed_dict={real_image_input: batch_x, noise_input: z, is_training: True})
        writer.add_summary(gdsum)

    if e % 500 == 0 or e == 1:
        print('Epoch %i: Generator Loss: %f, Discriminator Loss: %f' % (e, gl, dl))
        z = np.random.uniform(-1., 1., size=[1, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z, is_training:False})
        smilesganlist.append(g)

smileslist=list()
for i in range(len(smilesganlist)):
    temp = (smilesganlist[i]+1 )/2
    smileslist.append(np.reshape(temp, newshape=[28, 28]))


import csv

with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(smileslist)