# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

#((conv-bn-relu)*2-pool)*2-dense-bn-relu-dense
class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            # img 28x28x1 (black/white), Input Layer
            X = tf.reshape(self.X, [-1, 28, 28, 1])
            conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3])
            bn1 = tf.layers.batch_normalization(conv1, training=self.training)
            relu1 = tf.nn.relu(bn1)
            conv2 = tf.layers.conv2d(inputs=relu1, filters=64, kernel_size=[3, 3])
            bn2 = tf.layers.batch_normalization(conv2, training=self.training)
            relu2 = tf.nn.relu(bn2)
            pool1 = tf.layers.max_pooling2d(inputs=relu2, pool_size=[2, 2], strides=2)

            conv3 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3])
            bn3 = tf.layers.batch_normalization(conv3, training=self.training)
            relu3 = tf.nn.relu(bn3)
            conv4 = tf.layers.conv2d(inputs=relu3, filters=256, kernel_size=[3, 3])
            bn4 = tf.layers.batch_normalization(conv4, training=self.training)
            relu4 = tf.nn.relu(bn4)
            pool2 = tf.layers.max_pooling2d(inputs=relu4, pool_size=[2, 2], strides=2)

            input_affine = tf.reshape(pool2, [-1, 4 * 4 * 256])
            dense1 = tf.layers.dense(inputs=input_affine, units=1024)
            bn5 = tf.layers.batch_normalization(dense1, training=self.training)
            relu5 = tf.nn.relu(bn5)
            dense2 = tf.layers.dense(inputs=relu5, units=10, use_bias=True)
            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = dense2

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
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
#Accuracy: 0.3349
