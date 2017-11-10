import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import timeit
import time
import matplotlib.pyplot as plt
import os


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


lr = 0.001  # learning rate
BATCH_SIZE = 128
N_EPOCHS = 10
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
SKIP_STEP = 100
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


def my_model(X, Y, is_training):
    X = tf.reshape(X, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3])
    bn1 = tf.layers.batch_normalization(conv1, training=is_training)
    relu1 = tf.nn.relu(bn1)
    conv2 = tf.layers.conv2d(inputs=relu1, filters=64, kernel_size=[3, 3])
    bn2 = tf.layers.batch_normalization(conv2, training=is_training)
    relu2 = tf.nn.relu(bn2)
    pool1 = tf.layers.max_pooling2d(inputs=relu2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3])
    bn3 = tf.layers.batch_normalization(conv3, training=is_training)
    relu3 = tf.nn.relu(bn3)
    conv4 = tf.layers.conv2d(inputs=relu3, filters=256, kernel_size=[3, 3])
    bn4 = tf.layers.batch_normalization(conv4, training=is_training)
    relu4 = tf.nn.relu(bn4)
    pool2 = tf.layers.max_pooling2d(inputs=relu4, pool_size=[2, 2], strides=2)

    input_affine = tf.reshape(pool2, [-1, 4 * 4 * 256])
    dense1 = tf.layers.dense(inputs=input_affine, units=1024)
    bn5 = tf.layers.batch_normalization(dense1, training=is_training)
    relu5 = tf.nn.relu(bn5)
    dense2 = tf.layers.dense(inputs=relu5, units=10, use_bias=True)
    return dense2


y_out = my_model(X, Y, is_training)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_out))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
acc = tf.equal(tf.argmax(y_out, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))
tf.summary.scalar('loss', loss)
tf.summary.scalar('acc', acc)
summary_op = tf.summary.merge_all()

make_dir('checkpoints')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times
        print(index)
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch, summary = sess.run([optimizer, loss, summary_op],
                                          feed_dict={X: X_batch, Y: Y_batch, is_training: True})
        writer.add_summary(summary, global_step=index)
        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/mnist', index)

    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))
    print("Accuracy:", acc.eval({X: mnist.test.images, Y: mnist.test.labels}))
