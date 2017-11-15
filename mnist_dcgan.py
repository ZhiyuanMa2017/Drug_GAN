import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

num_steps = 10000
batch_size = 128
lr_generator = 0.002
lr_discriminator = 0.002

image_dim = 784
noise_dim = 100

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
        img = tf.layers.conv2d_transpose(layer9, filters=1, kernel_size=4, strides=2, padding='SAME',
                                         activation=tf.tanh)
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

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1, num_steps + 1):

    batch_x, _ = mnist.train.next_batch(batch_size)
    batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
    batch_x = batch_x * 2. - 1.

    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, dl = sess.run([train_disc, disc_loss], feed_dict={real_image_input: batch_x, noise_input: z, is_training: True})

    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, gl = sess.run([train_gen, gen_loss], feed_dict={noise_input: z, is_training: True})

    if i % 500 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

#test
n = 6
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    g = sess.run(gen_sample, feed_dict={noise_input: z, is_training:False})
    for j in range(n):
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])
plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()
plt.savefig("gantemp.jpg")