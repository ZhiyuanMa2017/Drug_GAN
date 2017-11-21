import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from load_cifar10 import load_CIFAR10

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

noise_dim = 96
show_every=250
print_every=50
batch_size=1024
num_epoch=100
# def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
#     cifar10_dir = 'data/cifar-10-batches-py'
#     X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
#     # Subsample the data
#     mask = range(num_training, num_training + num_validation)
#     X_val = X_train[mask]
#     y_val = y_train[mask]
#     mask = range(num_training)
#     X_train = X_train[mask]
#     y_train = y_train[mask]
#     mask = range(num_test)
#     X_test = X_test[mask]
#     y_test = y_test[mask]
#     # Normalize the data: subtract the mean image
#     mean_image = np.mean(X_train, axis=0)
#     X_train -= mean_image
#     X_val -= mean_image
#     X_test -= mean_image
#     return X_train, y_train, X_val, y_val, X_test, y_test

cifar10_dir = 'data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image

# Invoke the above function to get our data.
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)


def sample_noise(batch_size, dim):
    # return tf.random_uniform([batch_size, dim], minval=-1, maxval=1)
    return tf.random_normal([batch_size, dim])

def discriminator(x):
    with tf.variable_scope("discriminator"):
        x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='SAME', activation=leaky_relu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='SAME', activation=leaky_relu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2, padding='SAME', activation=leaky_relu)
        x = tf.layers.conv2d(x, filters=512, kernel_size=5, strides=2, padding='SAME', activation=leaky_relu)
        x = tf.reshape(x, [-1,2048])
        x = tf.layers.dense(x, units=1024, activation=leaky_relu)
        x = tf.layers.dense(x, units=1)
        return x


def generator(z):
    with tf.variable_scope("generator"):
        z = tf.layers.dense(z, units=2048)
        z = tf.layers.batch_normalization(z, training=is_training)
        z = tf.nn.relu(z)
        z = tf.reshape(z, [-1, 2, 2, 512])
        z = tf.layers.conv2d_transpose(z, filters=256, kernel_size=5, strides=2, padding='SAME')
        z = tf.layers.batch_normalization(z, training=is_training)
        z = tf.nn.relu(z)
        z = tf.layers.conv2d_transpose(z, filters=128, kernel_size=5, strides=2, padding='SAME')
        z = tf.layers.batch_normalization(z, training=is_training)
        z = tf.nn.relu(z)
        z = tf.layers.conv2d_transpose(z, filters=64, kernel_size=5, strides=2, padding='SAME')
        z = tf.layers.batch_normalization(z, training=is_training)
        z = tf.nn.relu(z)
        z = tf.layers.conv2d_transpose(z, filters=3, kernel_size=4, strides=2, padding='SAME',
                                         activation=tf.tanh)
        return z

tf.reset_default_graph()


def get_solvers(lr1=1e-3, lr2=1e-4, beta1=0.5):
    D_solver = tf.train.AdamOptimizer(learning_rate=lr1, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=lr2, beta1=beta1)
    pass
    return D_solver, G_solver

# our noise dimension


# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
z = sample_noise(batch_size, noise_dim)
is_training = tf.placeholder(tf.bool)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(x)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)



ones = tf.ones_like(logits_fake)
zeros = tf.zeros_like(logits_fake)
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=ones))
D_loss = 0
D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=ones))
D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=zeros))


# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')

D_solver,G_solver = get_solvers()

D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'generator')
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)
summ = tf.summary.merge_all()


sess=get_session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./graphs', sess.graph)
for epoch in range(num_epoch):
    total_batch = int(len(X_train) / batch_size)
    for i in range(total_batch):
        batch_x = X_train[i:i + batch_size]
        _,  __= sess.run([D_train_step,G_train_step], feed_dict={x: batch_x,is_training: True})
        D_loss_curr,  G_loss_curr, summm = sess.run([D_loss, G_loss, summ], feed_dict={x: batch_x,is_training: True})
        # _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={is_training: True})
        writer.add_summary(summm)
    # print loss every so often.
    # We want to make sure D_loss doesn't go to 0
    print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
    samples = sess.run(G_sample, feed_dict={is_training: False})
    samples = (samples + 1) / 2
    for h in range(10):
        plt.subplot(1,10,h+1)
        plt.imshow(samples[h].astype('uint8'))
    plt.savefig('ep' + str(epoch) + '.jpg')

print('Final images')
samples = sess.run(G_sample,feed_dict={is_training: False})
samples = (samples + 1) / 2
for h in range(10):
    plt.subplot(1,10,h+1)
    plt.imshow(samples[h].astype('uint8'))
plt.savefig('cifarfinal' + '.jpg')