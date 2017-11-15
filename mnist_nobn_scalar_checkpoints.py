import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 128


X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None,10])
is_training = tf.placeholder(tf.bool)

#((conv-relu)*2-pool)*2-dense-relu-dense
def my_model(X, Y, is_training):
    X_img = tf.reshape(X, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    input_affine = tf.reshape(pool2, [-1, 4 * 4 * 256])
    dense1 = tf.layers.dense(inputs=input_affine, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense1, units=10, use_bias=True)
    return logits

logits = my_model(X,Y,is_training)
loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('loss', loss)
tf.summary.scalar('acc', accuracy)
sum = tf.summary.merge_all()

# initialize
make_dir('checkpoints')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Learning Started!')
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_acc = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _ , c, summ = sess.run([optimizer, loss, sum], feed_dict={X: batch_xs, Y: batch_ys, is_training:True})
            writer.add_summary(summ)
            avg_cost += float(c)/ total_batch
            a = sess.run(accuracy,feed_dict={X: batch_xs, Y: batch_ys, is_training:False})
            avg_acc += a/ total_batch
            if epoch == training_epochs-1 and i == total_batch-1:
                index = epoch*total_batch+i
                saver.save(sess, 'checkpoints/mnist', index)
#尝试 未运行实现
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_cost),'acc =', '{:.9f}'.format(avg_acc))

print('Learning Finished!')

# Test model and check accuracy
#print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
#final accuracy:0.9937
#

