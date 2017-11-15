from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# A bunch of utility functions

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

from read_mnist import load_train_images,load_train_labels,load_test_images,load_test_labels
train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()
temp = train_labels.astype(int)
temp2 = test_labels.astype(int)
# np.bincount(temp)
mask = np.where(temp==1)
trainone = train_images[mask]
mask2 = np.where(temp2==1)
testone = test_images[mask2]
# for i in range(10):
#     plt.imshow(testone[i], cmap='gray')
#     plt.savefig('one'+str(i)+"jpg")
numone_image = np.concatenate((trainone,testone))


def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)


def sample_noise(batch_size, dim):
    return tf.random_uniform([batch_size, dim], minval=-1, maxval=1)

def discriminator(x):
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        x = tf.reshape(x, [-1, 28, 28, 1])
        layer1 = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, activation=leaky_relu)
        layer2 = tf.layers.max_pooling2d(layer1, pool_size=2, strides=2)
        layer3 = tf.layers.conv2d(layer2, filters=64, kernel_size=5, strides=1, activation=leaky_relu)
        layer4 = tf.layers.max_pooling2d(layer3, pool_size=2, strides=2)
        x_flat = tf.contrib.layers.flatten(layer4)
        layer5 = tf.layers.dense(x_flat, units=4 * 4 * 64, activation=leaky_relu)
        logits = tf.layers.dense(layer5, units=1)
        return logits


def generator(z):
    with tf.variable_scope("generator"):
        # TODO: implement architecture
        layer1 = tf.layers.dense(z, units=1024, activation=tf.nn.relu)
        layer2 = tf.layers.batch_normalization(layer1, training=True)
        layer3 = tf.layers.dense(layer2, units=7 * 7 * 128, activation=tf.nn.relu)
        layer4 = tf.layers.batch_normalization(layer3, training=True)
        imgresize = tf.reshape(layer4, [-1, 7, 7, 128])
        layer5 = tf.layers.conv2d_transpose(imgresize, filters=64, kernel_size=4, strides=2, padding='SAME',
                                            activation=tf.nn.relu)
        layer6 = tf.layers.batch_normalization(layer5, training=True)
        img = tf.layers.conv2d_transpose(layer6, filters=1, kernel_size=4, strides=2, padding='SAME',
                                         activation=tf.tanh)
        return img

tf.reset_default_graph()


def get_solvers(learning_rate=1e-3, beta1=0.5):
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    pass
    return D_solver, G_solver

batch_size = 128
# our noise dimension
noise_dim = 96

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 28, 28])
z = sample_noise(batch_size, noise_dim)
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

show_every=250
print_every=50
batch_size=128
num_epoch=10
with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epoch):
        total_batch = int(len(numone_image) / batch_size)
        for i in range(total_batch):
            batch_x = numone_image[i:i + batch_size]
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: batch_x})
            # for j in range(1000):
            #     _, = sess.run([G_train_step])
            # _, G_loss_curr = sess.run([G_train_step, G_loss])
        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        samples = sess.run(G_sample)
        fig = show_images(samples[:16])
        plt.savefig('ep' + str(epoch) + 'jpg')
        print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))

    print('Final images')
    samples = sess.run(G_sample)
    fig = show_images(samples[:16])
    plt.savefig('finaltempjpg')