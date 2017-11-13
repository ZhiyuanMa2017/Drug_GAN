import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)


# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 128

#((conv-relu)*2-pool)*2-dense-relu-dense
class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.is_training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)

            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            conv3 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], activation=tf.nn.relu)

            conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

            input_affine = tf.reshape(pool2, [-1, 4 * 4 * 256])

            dense1 = tf.layers.dense(inputs=input_affine, units=1024, activation=tf.nn.relu)

            self.logits = tf.layers.dense(inputs=dense1, units=10, use_bias=True)


        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.is_training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.is_training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.is_training: training})



# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())

print('Learning Started!')
writer = tf.summary.FileWriter('./graphs', sess.graph)

# train my model
loss = []
acc = []
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_acc = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch
        a = m1.get_accuracy(batch_xs, batch_ys)
        avg_acc += a / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_cost),'acc =', '{:.9f}'.format(avg_acc))
    loss.append(avg_cost)
    acc.append(avg_acc)

#
print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
#final accuracy:0.9937


