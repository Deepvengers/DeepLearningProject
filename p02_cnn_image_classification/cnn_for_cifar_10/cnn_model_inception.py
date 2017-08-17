import tensorflow as tf
import numpy as np
import time
from tensorflow.python.training import moving_averages as ema



class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                self.training = tf.placeholder(tf.bool, name='training')

                self.X = tf.placeholder(tf.float32, [None, 1024], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 32, 32, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10], name='y_data')

            with tf.name_scope('conv_layer1') as scope:
                # inception module
                self.L1_sub = self.inception2d_11_33_55_maxpool11_1(X_img)
                self.L1 = tf.nn.max_pool(value=self.L1_sub, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('conv_layer2') as scope:
                self.L2 = self.inception2d_11_33_55_maxpool11_2(self.L1)
                self.L2 = tf.nn.max_pool(value=self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('conv_layer3') as scope:
                self.W3_sub = tf.get_variable(name='W3_sub', shape=[3, 3, 160, 160], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_sub = tf.nn.conv2d(input=self.L2, filter=self.W3_sub, strides=[1, 1, 1, 1], padding='VALID')
                self.L3_sub = self.parametric_relu(self.L3_sub, 'R3_sub')
                self.W4_sub = tf.get_variable(name='W4_sub', shape=[3, 3, 160, 160], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4_sub = tf.nn.conv2d(input=self.L3_sub, filter=self.W4_sub, strides=[1, 1, 1, 1], padding='VALID')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R4_sub')
                self.W5_sub = tf.get_variable(name='W5_sub', shape=[3, 3, 160, 160], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L5_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.W5_sub, strides=[1, 1, 1, 1], padding='SAME')
                # self.L5_sub = self.batch_norm(self.L5_sub, shape=self.L5_sub.get_shape()[-1], training=self.training, convl=True, name='BN3-3')
                self.L5_sub = self.batch_norm(input=self.L5_sub, shape=160, training=self.training, convl=True, name='BN3-3')
                self.L5_sub = self.parametric_relu(self.L5_sub, 'R5_sub')

            with tf.name_scope('conv_layer4') as scope:
                self.W4 = tf.get_variable(name='W4', shape=[3, 3, 160, 320], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4 = tf.nn.conv2d(input=self.L5_sub, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME')
                # self.L4 = self.batch_norm(self.L4, shape=self.L4.get_shape()[-1], training=self.training, convl=True, name='BN4')
                self.L4 = self.batch_norm(input=self.L4, shape=320, training=self.training, convl=True, name='BN4')
                self.L4 = self.parametric_relu(self.L4, 'R4')

            with tf.name_scope('conv_layer5') as scope:
                self.W6_sub = tf.get_variable(name='W6_sub', shape=[1, 3, 320, 320], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L6_sub = tf.nn.conv2d(input=self.L4, filter=self.W6_sub, strides=[1, 1, 1, 1], padding='SAME')
                self.L6_sub = self.parametric_relu(self.L6_sub, 'R6_sub')
                self.W7_sub = tf.get_variable(name='W7_sub', shape=[3, 1, 320, 640], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L7_sub = tf.nn.conv2d(input=self.L6_sub, filter=self.W7_sub, strides=[1, 1, 1, 1], padding='SAME')
                # self.L7_sub = self.batch_norm(self.L7_sub, shape=self.L7_sub.get_shape()[-1], training=self.training, convl=True, name='BN5-2')
                self.L7_sub = self.batch_norm(input=self.L7_sub, shape=640, training=self.training, convl=True, name='BN5-2')
                self.L7_sub = self.parametric_relu(self.L7_sub, 'R7_sub')
                self.L5 = tf.reshape(self.L7_sub, shape=[-1, 4 * 4 * 640])

            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[4 * 4 * 640, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc1'))
                self.L6 = tf.matmul(self.L5, self.W_fc1)
                # self.L_fc1 = self.batch_norm(self.L5, shape=self.L5.get_shape()[-1], training=self.training, convl=False, name='BN6')
                self.L_fc1 = self.batch_norm(input=self.L6, shape=1000, training=self.training, convl=False, name='BN6')
                # self.L_fc1 = self.parametric_relu(tf.matmul(self.L5, self.W_fc1), 'R_fc1')
                self.L_fc1 = self.parametric_relu(self.L6, 'R_fc1')

            with tf.name_scope('fc_layer2') as scope:
                self.W_fc2 = tf.get_variable(name='W_fc2', shape=[1000, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc2'))
                self.L7 = tf.matmul(self.L_fc1, self.W_fc2)
                # self.L_fc2 = self.batch_norm(self.L_fc1, shape=self.L_fc1.get_shape()[-1], training=self.training, convl=False, name='BN7')
                self.L_fc2 = self.batch_norm(self.L7, shape=1000, training=self.training, convl=False, name='BN7')
                # self.L_fc2 = self.parametric_relu(tf.matmul(self.L_fc1, self.W_fc2), 'R_fc2')
                self.L_fc2 = self.parametric_relu(self.L7, 'R_fc2')

            self.W_out = tf.get_variable(name='W_out', shape=[1000, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[10], name='b_out'))
            self.logits = tf.matmul(self.L_fc2, self.W_out) + self.b_out

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + ( 0.01 / ( 2 * tf.to_float( tf.shape(self.Y)[0]))) * tf.reduce_sum(tf.square(self.W_out))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def batch_norm(self, input, shape, training, convl=True, name='BN'):
        beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
        scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
        if convl:
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        else:
            batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)

    def inception2d_11_33_55_maxpool11_1(self, x):
        self.b1 = tf.Variable(tf.constant(value=0.001, shape=[40], name='b1'))

        # 1x1
        self.W1_1 = tf.get_variable(name='W1_1', shape=[1, 1, 1, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_1 = tf.nn.conv2d(input=x, filter=self.W1_1, strides=[1, 1, 1, 1], padding='SAME')

        # 5x5 -> 1x3 3x1 1x3 3x1
        self.W1_2_1x3_1_sub = tf.get_variable(name='W1_2_1x3_1', shape=[1, 3, 1, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_2_1_sub = tf.nn.conv2d(input=x, filter=self.W1_2_1x3_1_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_2_3x1_1_sub = tf.get_variable(name='W1_2_3x1_1', shape=[3, 1, 10, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_2_2_sub = tf.nn.conv2d(input=self.L1_2_1_sub, filter=self.W1_2_3x1_1_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_2_1x3_2_sub = tf.get_variable(name='W1_2_1x3_2', shape=[1, 3, 10, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_2_3_sub = tf.nn.conv2d(input=self.L1_2_2_sub, filter=self.W1_2_1x3_2_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_2_3x1_2_sub = tf.get_variable(name='W1_2_3x1_2', shape=[3, 1, 10, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_2 = tf.nn.conv2d(input=self.L1_2_3_sub, filter=self.W1_2_3x1_2_sub, strides=[1, 1, 1, 1], padding='SAME')

        # 3x3 -> 1x3 3x1
        self.W1_3_1x3_sub = tf.get_variable(name='W1_3_1x3', shape=[1, 3, 1, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_3_sub = tf.nn.conv2d(input=x, filter=self.W1_3_1x3_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_3_3x1_sub = tf.get_variable(name='W1_3_3x1', shape=[3, 1, 10, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_3 = tf.nn.conv2d(input=self.L1_3_sub, filter=self.W1_3_3x1_sub, strides=[1, 1, 1, 1], padding='SAME')

        # max pooling + 1x1
        self.L1_4 = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.W1_4 = tf.get_variable(name='W1_4', shape=[1, 1, 1, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_4 = tf.nn.conv2d(input=self.L1_4, filter=self.W1_4, strides=[1, 1, 1, 1], padding='SAME')

        x = tf.concat([self.L1_1, self.L1_2, self.L1_3, self.L1_4], axis=3)  # Concat in the 4th dim to stack
        self.L1 = self.batch_norm(x, shape=x.get_shape()[-1], training=self.training, convl=True,name='inception_BN1')
        return self.parametric_relu(x + self.b1, 'L1')

    def inception2d_11_33_55_maxpool11_2(self, x):
        self.b2 = tf.Variable(tf.constant(value=0.001, shape=[160], name='b2'))

        # 1x1
        self.W2_1 = tf.get_variable(name='W2_1', shape=[1, 1, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L2_1 = tf.nn.conv2d(input=x, filter=self.W2_1, strides=[1, 1, 1, 1], padding='SAME')

        # 5x5 -> 1x3 3x1 1x3 3x1
        self.W2_2_1x3_1_sub = tf.get_variable(name='W2_2_1x3_1', shape=[1, 3, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L2_2_1_sub = tf.nn.conv2d(input=x, filter=self.W2_2_1x3_1_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W2_2_3x1_1_sub = tf.get_variable(name='W2_2_3x1_1', shape=[3, 1, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L2_2_2_sub = tf.nn.conv2d(input=self.L2_2_1_sub, filter=self.W2_2_3x1_1_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W2_2_1x3_2_sub = tf.get_variable(name='W2_2_1x3_2', shape=[1, 3, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L2_2_3_sub = tf.nn.conv2d(input=self.L2_2_2_sub, filter=self.W2_2_1x3_2_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W2_2_3x1_2_sub = tf.get_variable(name='W2_2_3x1_2', shape=[3, 1, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L2_2 = tf.nn.conv2d(input=self.L2_2_3_sub, filter=self.W2_2_3x1_2_sub, strides=[1, 1, 1, 1], padding='SAME')

        # 3x3 -> 1x3 3x1
        self.W2_3_1x3_sub = tf.get_variable(name='W2_3_1x3', shape=[1, 3, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L2_3_sub = tf.nn.conv2d(input=x, filter=self.W2_3_1x3_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W2_3_3x1_sub = tf.get_variable(name='W2_3_3x1', shape=[3, 1, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L2_3 = tf.nn.conv2d(input=self.L2_3_sub, filter=self.W2_3_3x1_sub, strides=[1, 1, 1, 1], padding='SAME')

        # max pooling
        self.L2_4 = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.W2_4 = tf.get_variable(name='W2_4', shape=[1, 1, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L2_4 = tf.nn.conv2d(input=self.L2_4, filter=self.W2_4, strides=[1, 1, 1, 1], padding='SAME')

        x = tf.concat([self.L2_1, self.L2_2, self.L2_3, self.L2_4], axis=3)  # Concat in the 4th dim to stack
        self.L2 = self.batch_norm(x, shape=x.get_shape()[-1], training=self.training, convl=True, name='inception_BN2')
        return self.parametric_relu(x + self.b2, 'L2')
