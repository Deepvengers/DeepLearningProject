import tensorflow as tf
import numpy as np
import re

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.inception_reg_losses = [0., 0., 0.]
        self.ac_optimizer = {}
        self.early_stop_count = 0
        self.epoch = 0
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')

                self.X = tf.placeholder(tf.float32, [None, 1024], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 32, 32, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10], name='y_data')

            with tf.name_scope('stem_layer') as scope:
                self.W1_sub = tf.get_variable(name='W1_sub', shape=[1, 3, 1, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_sub = tf.nn.conv2d(input=X_img, filter=self.W1_sub, strides=[1, 1, 1, 1], padding='VALID')  # 32x32 -> 32x30
                self.L1_sub = self.BN(input=self.L1_sub, training=self.training, name='L1_sub_BN')
                self.L1_sub = self.parametric_relu(self.L1_sub, 'R1_sub')
                self.W2_sub = tf.get_variable(name='W2_sub', shape=[3, 1, 30, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2_sub = tf.nn.conv2d(input=self.L1_sub, filter=self.W2_sub, strides=[1, 1, 1, 1], padding='VALID')  # 32x30 -> 30x30
                self.L2_sub = self.BN(input=self.L2_sub, training=self.training, name='L2_sub_BN')
                self.L2_sub = self.parametric_relu(self.L2_sub, 'R2_sub')
                self.W3_sub = tf.get_variable(name='W3_sub', shape=[1, 3, 30, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W3_sub, strides=[1, 1, 1, 1], padding='VALID')  # 30x30 -> 30x28
                self.L3_sub = self.BN(input=self.L3_sub, training=self.training, name='L3_sub_BN')
                self.L3_sub = self.parametric_relu(self.L3_sub, 'R3_sub')
                self.W4_sub = tf.get_variable(name='W4_sub', shape=[3, 1, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4_sub = tf.nn.conv2d(input=self.L3_sub, filter=self.W4_sub, strides=[1, 1, 1, 1], padding='VALID')  # 30x28 -> 28x28
                self.L4_sub = self.BN(input=self.L4_sub, training=self.training, name='L4_sub_BN')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R4_sub')
                # self.W1_sub = tf.get_variable(name='W1_sub', shape=[5, 5, 1, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.L1_sub = tf.nn.conv2d(input=X_img, filter=self.W1_sub, strides=[1, 1, 1, 1], padding='VALID')  # 32x32 -> 32x30
                # self.L1_sub = self.BN(input=self.L1_sub, training=self.training, name='L1_sub_BN')
                # self.L1_sub = self.parametric_relu(self.L1_sub, 'R1_sub')
                self.W5_sub = tf.get_variable(name='W5_sub', shape=[1, 3, 40, 50], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L5_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.W5_sub, strides=[1, 1, 1, 1], padding='VALID')  # 28x28 -> 28x26
                self.L5_sub = self.BN(input=self.L5_sub, training=self.training, name='L5_sub_BN')
                self.L5_sub = self.parametric_relu(self.L5_sub, 'R5_sub')
                self.W6_sub = tf.get_variable(name='W6_sub', shape=[3, 1, 50, 50], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L6_sub = tf.nn.conv2d(input=self.L5_sub, filter=self.W6_sub, strides=[1, 1, 1, 1], padding='VALID')  # 28x26 -> 26x26
                self.L6_sub = self.BN(input=self.L6_sub, training=self.training, name='L6_sub_BN')
                self.L6_sub = self.parametric_relu(self.L6_sub, 'R6_sub')
                # self.L1 = tf.nn.max_pool(value=self.L3_sub, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 26x26 -> 13x13

            with tf.name_scope('inception_layer1') as scope:
                self.Inception_L1 = self.inception_A(self.L6_sub, 3, 200, name='inception_layer1')  # 26x26 -> 13x13

            with tf.name_scope('inception_layer2') as scope:
                self.Inception_L2 = self.inception_B(self.Inception_L1, 3, 250, name='inception_layer2')  # 13x13 -> 7x7
                self.auxiliary_classifiers(self.Inception_L2, 300, 700, 700, 'ac1')

            with tf.name_scope('inception_layer3') as scope:
                self.Inception_L3 = self.inception_C(self.Inception_L2, 3, 300, name='inception_layer3')  # 7x7 -> 4x4
                # self.Inception_L3 = tf.layers.dropout(inputs=self.Inception_L3, rate=self.dropout_rate, training=self.training)
                # self.Inception_L3 = tf.reshape(self.Inception_L3, shape=[-1, 4 * 4 * 200])

            with tf.name_scope('conv_layer1') as scope:
                self.W4 = tf.get_variable(name='W4', shape=[3, 3, 300, 350], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4 = tf.nn.conv2d(input=self.Inception_L3, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME')
                self.L4 = self.BN(input=self.L4, training=self.training, name='conv1_BN')
                self.L4 = self.parametric_relu(self.L4, 'R4')
                # self.L4 = tf.reshape(self.L4, shape=[-1, 4 * 4 * 240])
                # self.L4 = tf.layers.dropout(inputs=self.L4, rate=self.dropout_rate, training=self.training)

            with tf.name_scope('conv_layer2') as scope:
                self.W5 = tf.get_variable(name='W5', shape=[3, 3, 350, 400], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L5 = tf.nn.conv2d(input=self.L4, filter=self.W5, strides=[1, 1, 1, 1], padding='SAME')
                self.L5 = self.BN(input=self.L5, training=self.training, name='conv2_BN')
                self.L5 = self.parametric_relu(self.L5, 'R5')
                # self.L5 = tf.nn.max_pool(value=self.L5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
                self.L5 = tf.reshape(self.L5, shape=[-1, 4 * 4 * 400])
                # self.L5 = tf.layers.dropout(inputs=self.L5, rate=self.dropout_rate, training=self.training)

            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[4 * 4 * 400, 850], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[850], name='b_fc1'))
                self.L_fc1 = tf.matmul(self.L5, self.W_fc1) + self.b_fc1
                self.L_fc1 = self.BN(input=self.L_fc1, training=self.training, name='fc1_BN')
                self.L_fc1 = self.parametric_relu(self.L_fc1, 'R_fc1')
                self.L_fc1 = tf.layers.dropout(inputs=self.L_fc1, rate=self.dropout_rate, training=self.training)

            # with tf.name_scope('fc_layer2') as scope:
            #     self.W_fc2 = tf.get_variable(name='W_fc2', shape=[1000, 500], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            #     self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[500], name='b_fc2'))
            #     self.L_fc2 = tf.matmul(self.L_fc1, self.W_fc2) + self.b_fc2
            #     self.L_fc2 = self.BN(input=self.L_fc2, training=self.training, name='fc2_BN')
            #     self.L_fc2 = self.parametric_relu(self.L_fc2, 'R_fc2')
            #     # self.L_fc2 = tf.layers.dropout(inputs=self.L_fc2, rate=self.dropout_rate, training=self.training)

            with tf.name_scope('fc_layer3') as scope:
                self.W_fc3 = tf.get_variable(name='W_fc3', shape=[850, 850], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc3 = tf.Variable(tf.constant(value=0.001, shape=[850], name='b_fc3'))
                self.L_fc3 = tf.matmul(self.L_fc1, self.W_fc3) + self.b_fc3
                self.L_fc3 = self.BN(input=self.L_fc3, training=self.training, name='fc3_BN')
                self.L_fc3 = self.parametric_relu(self.L_fc3, 'R_fc3')
                self.L_fc3 = tf.layers.dropout(inputs=self.L_fc3, rate=self.dropout_rate, training=self.training)

            self.W_out = tf.get_variable(name='W_out', shape=[850, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[10], name='b_out'))
            self.logits = tf.matmul(self.L_fc3, self.W_out) + self.b_out

            self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(train_var) for train_var in tf.get_variable_scope().trainable_variables() if re.search(self.name+'\/W.*', train_var.name) is not None]) + \
                            tf.reduce_sum(self.inception_reg_losses)
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + 0.0001 * self.reg_loss
            self.cost = self.cost + (0.3 * self.ac_cost)

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.dynamic_learning(0.005, self.early_stop_count, self.epoch)).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    ################################################################################################################
    ## ▣ auxiliary_classifiers - Created by 이용은
    ## - x : input
    ## - n1, n2 : fc 1, 2차 output 갯수
    ## - name : 보조 분류기 간의 변수명 충돌을 막기 위한 고유 이름
    ## - learning_rate : 해당 보조 분류기에서 적용할 learning rate
    ################################################################################################################
    def auxiliary_classifiers(self, input, convl_out, fc1_out, fc2_out, name):
        with tf.variable_scope(name):
            B, H, W, C = input.get_shape().as_list()

            with tf.name_scope(name + '_conv_layer1') as scope:
                ac_W1x1 = tf.get_variable(name = name + '_W1x1', shape=[1, 1, C, C], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                ac_L1x1 = tf.nn.conv2d(input=input, filter=ac_W1x1, strides=[1, 1, 1, 1], padding='SAME', name=name + '_L3x3') # 7x7 -> 7x7
                ac_L1x1 = self.BN(input=ac_L1x1, training=self.training, name=name + '_L1x1_BN')
                ac_L1x1 = self.parametric_relu(ac_L1x1, name=name + '_L1x1_R')

            with tf.name_scope(name + '_conv_layer2') as scope:
                ac_W3x3 = tf.get_variable(name=name + '_W3x3', shape=[3, 3, C, convl_out], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                ac_L3x3 = tf.nn.conv2d(input=ac_L1x1, filter=ac_W3x3, strides=[1, 1, 1, 1], padding='SAME', name=name + '_L3x3') # 4x4 -> 4x4
                ac_L3x3 = self.BN(input=ac_L3x3, training=self.training, name=name + '_L3x3_BN')
                ac_L3x3 = self.parametric_relu(ac_L3x3, name=name + '_L3x3_R')
                ac_L3x3 = tf.nn.max_pool(value=ac_L3x3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name + '_maxppol') # 4x4 -> 1x1
                ac_L3x3 = tf.reshape(ac_L3x3, shape=[-1, 4 * 4 * convl_out])

            with tf.name_scope(name + '_fc_layer1') as scope:
                ac_W_fc1 = tf.get_variable(name=name + '_W_fc1', shape=[4 * 4 * convl_out, fc1_out], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                ac_b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[fc1_out], name=name + '_b_fc1'))
                ac_L_fc1 = tf.matmul(ac_L3x3, ac_W_fc1) + ac_b_fc1
                ac_L_fc1 = self.BN(input=ac_L_fc1, training=self.training, name=name + '_fc1_BN')
                ac_L_fc1 = self.parametric_relu(ac_L_fc1, name + '_R_fc1')
                ac_L_fc1 = tf.layers.dropout(inputs=ac_L_fc1, rate=self.dropout_rate, training=self.training)

            with tf.name_scope(name + 'fc_layer2') as scope:
                ac_W_fc2 = tf.get_variable(name=name + '_W_fc2', shape=[fc1_out, fc2_out], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                ac_b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[fc2_out], name=name + '_b_fc2'))
                ac_L_fc2 = tf.matmul(ac_L_fc1, ac_W_fc2) + ac_b_fc2
                ac_L_fc2 = self.BN(input=ac_L_fc2, training=self.training, name=name + '_fc2_BN')
                ac_L_fc2 = self.parametric_relu(ac_L_fc2, name + '_R_fc2')
                ac_L_fc2 = tf.layers.dropout(inputs=ac_L_fc2, rate=self.dropout_rate, training=self.training)

            ac_W_out = tf.get_variable(name=name + '_W_out', shape=[fc2_out, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            ac_b_out = tf.Variable(tf.constant(value=0.001, shape=[10], name=name + '_b_out'))
            ac_logits = tf.matmul(ac_L_fc2, ac_W_out) + ac_b_out

            stem_loss = tf.reduce_sum([tf.nn.l2_loss(train_var) for train_var in tf.trainable_variables() if re.search(self.name+'\/W\d\_sub', train_var.name)])  # stem layer L2 regularization
            reg_loss = stem_loss + self.inception_reg_losses[0] + self.inception_reg_losses[1] + tf.reduce_sum([tf.nn.l2_loss(train_var) for train_var in tf.get_variable_scope().trainable_variables() if re.search('W', train_var.name)])
            self.ac_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ac_logits, labels=self.Y)) + 0.0001 * reg_loss

            # self.ac_optimizer[name] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(0.8*ac_cost)
            # self.ac_optimizer[name] = tf.train.AdamOptimizer(learning_rate=self.dynamic_learning(0.005, self.early_stop_count, self.epoch)).minimize(ac_cost)

    def dynamic_learning(self,learning_rate,earlystop,epoch):
        max_learning_rate = learning_rate
        min_learning_rate = max_learning_rate / 5
        learning_decay = 60 # 낮을수록 빨리 떨어진다.
        if earlystop >= 1:
            lr = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-epoch / learning_decay)
        else:
            lr = max_learning_rate
        return round(lr, 4)

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        job = [self.cost, self.optimizer]
        for opt in self.ac_optimizer.values():
            job.append(opt)
        return self.sess.run(job, feed_dict={self.X: x_data, self.Y: y_data, self.training: True})
        # return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def BN(self, input, training, name):
        return tf.contrib.layers.batch_norm(input, decay=0.99, scale=True, is_training=training, updates_collections=None, scope=name)

    def inception_A(self, x, n, output, name):
        OPL = int(output/4)
        B, H, W, C = x.get_shape()

        with tf.variable_scope(name):
            # 1x1
            W1x1 = tf.get_variable(name='W1x1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L1x1 = tf.nn.conv2d(name='L1x1', input=x, filter=W1x1, strides=[1, 2, 2, 1], padding='SAME')
            L1x1 = self.BN(input=L1x1, training=self.training, name='inceptionA_L1x1_BN')
            L1x1 = self.parametric_relu(L1x1, 'inceptionA_L1x1_R')

            # 5x5 -> 1x1, 3x3, 3x3
            W5x5_sub1 = tf.get_variable(name='W5x5_sub1', shape=[1, 1, C, 25], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub1 = tf.nn.conv2d(name='L5x5_sub1', input=x, filter=W5x5_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub1 = self.BN(input=L5x5_sub1, training=self.training, name='inceptionA_L5x5_sub1_BN')
            L5x5_sub1 = self.parametric_relu(L5x5_sub1, 'inceptionA_L5x5_sub1_R')

            W5x5_sub2 = tf.get_variable(name='W5x5_sub2', shape=[n, n, 25, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub2 = tf.nn.conv2d(name='L5x5_sub2', input=L5x5_sub1, filter=W5x5_sub2, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub2 = self.BN(input=L5x5_sub2, training=self.training, name='inceptionA_L5x5_sub2_BN')
            L5x5_sub2 = self.parametric_relu(L5x5_sub2, 'inceptionA_L5x5_sub2_R')

            W5x5_sub3 = tf.get_variable(name='W5x5_sub3', shape=[n, n, OPL, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub3 = tf.nn.conv2d(name='L5x5_sub3', input=L5x5_sub2, filter=W5x5_sub3, strides=[1, 2, 2, 1], padding='SAME')
            L5x5_sub3 = self.BN(input=L5x5_sub3, training=self.training, name='inceptionA_L5x5_sub3_BN')
            L5x5_sub3 = self.parametric_relu(L5x5_sub3, 'inceptionA_L5x5_sub3_R')

            # 3x3 -> 1x1, 3x3
            W3x3_sub1 = tf.get_variable(name='W3x3_sub1', shape=[1, 1, C, 25], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub1 = tf.nn.conv2d(name='L3x3_sub1', input=x, filter=W3x3_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L3x3_sub1 = self.BN(input=L3x3_sub1, training=self.training, name='inceptionA_L3x3_sub1_BN')
            L3x3_sub1 = self.parametric_relu(L3x3_sub1, 'inceptionA_L3x3_sub1_R')

            W3x3_sub2 = tf.get_variable(name='W3x3_sub2', shape=[n, n, 25, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub2 = tf.nn.conv2d(name='L3x3_sub2', input=L3x3_sub1, filter=W3x3_sub2, strides=[1, 2, 2, 1], padding='SAME')
            L3x3_sub2 = self.BN(input=L3x3_sub2, training=self.training, name='inceptionA_L3x3_sub2_BN')
            L3x3_sub2 = self.parametric_relu(L3x3_sub2, 'inceptionA_L3x3_sub2_R')

            # avg pooling -> avg pooling, 1x1
            L_pool = tf.nn.max_pool(name='L_pool', value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            W_pool_sub1 = tf.get_variable(name='W_pool_sub1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L_pool_sub1 = tf.nn.conv2d(name='L_pool_sub1', input=L_pool, filter=W_pool_sub1, strides=[1, 2, 2, 1], padding='SAME')
            L_pool_sub1 = self.BN(input=L_pool_sub1, training=self.training, name='inceptionA_L_pool_sub1_BN')
            L_pool_sub1 = self.parametric_relu(L_pool_sub1, 'inceptionA_L_pool_sub1_R')

            tot_layers = tf.concat([L1x1, L5x5_sub3, L3x3_sub2, L_pool_sub1], axis=3)  # Concat in the 4th dim to stack
            self.inception_reg_losses[0] = tf.reduce_sum([tf.nn.l2_loss(train_var) for train_var in tf.get_variable_scope().trainable_variables() if re.search('W', train_var.name) is not None])
        return tot_layers

    def inception_B(self, x, n, output, name):
        B, H, W, C = x.get_shape()

        with tf.variable_scope(name):
            # 1x1
            W1x1 = tf.get_variable(name='W1x1', shape=[1, 1, C, 80], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L1x1 = tf.nn.conv2d(name='L1x1', input=x, filter=W1x1, strides=[1, 2, 2, 1], padding='SAME')
            L1x1 = self.BN(input=L1x1, training=self.training, name='inceptionB_L1x1_BN')
            L1x1 = self.parametric_relu(L1x1, 'inceptionB_L1x1_R')

            # 5x5 -> 1x1, 1x3, 3x1, 1x3, 3x1
            W5x5_sub1 = tf.get_variable(name='W5x5_sub1', shape=[1, 1, C, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub1 = tf.nn.conv2d(name='L5x5_sub1', input=x, filter=W5x5_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub1 = self.BN(input=L5x5_sub1, training=self.training, name='inceptionB_L5x5_sub1_BN')
            L5x5_sub1 = self.parametric_relu(L5x5_sub1, 'inceptionB_L5x5_sub1_R')

            W5x5_sub2 = tf.get_variable(name='W5x5_sub2', shape=[1, n, 30, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub2 = tf.nn.conv2d(name='L5x5_sub2', input=L5x5_sub1, filter=W5x5_sub2, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub2 = self.BN(input=L5x5_sub2, training=self.training, name='inceptionB_L5x5_sub2_BN')
            L5x5_sub2 = self.parametric_relu(L5x5_sub2, 'inceptionB_L5x5_sub2_R')

            W5x5_sub3 = tf.get_variable(name='W5x5_sub3', shape=[n, 1, 40, 50], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub3 = tf.nn.conv2d(name='L5x5_sub3', input=L5x5_sub2, filter=W5x5_sub3, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub3 = self.BN(input=L5x5_sub3, training=self.training, name='inceptionB_L5x5_sub3_BN')
            L5x5_sub3 = self.parametric_relu(L5x5_sub3, 'inceptionB_L5x5_sub3_R')

            W5x5_sub4 = tf.get_variable(name='W5x5_sub4', shape=[1, n, 50, 60], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub4 = tf.nn.conv2d(name='L5x5_sub4', input=L5x5_sub3, filter=W5x5_sub4, strides=[1, 1, 2, 1], padding='SAME')
            L5x5_sub4 = self.BN(input=L5x5_sub4, training=self.training, name='inceptionB_L5x5_sub4_BN')
            L5x5_sub4 = self.parametric_relu(L5x5_sub4, 'inceptionB_L5x5_sub4_R')

            W5x5_sub5 = tf.get_variable(name='W5x5_sub5', shape=[n, 1, 60, 70], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub5 = tf.nn.conv2d(name='L5x5_sub5', input=L5x5_sub4, filter=W5x5_sub5, strides=[1, 2, 1, 1], padding='SAME')
            L5x5_sub5 = self.BN(input=L5x5_sub5, training=self.training, name='inceptionB_L5x5_sub5_BN')
            L5x5_sub5 = self.parametric_relu(L5x5_sub5, 'inceptionB_L5x5_sub5_R')

            # 3x3 -> 1x1, 1x3, 3x1
            W3x3_sub1 = tf.get_variable(name='W3x3_sub1', shape=[1, 1, C, 50], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub1 = tf.nn.conv2d(name='L3x3_sub1', input=x, filter=W3x3_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L3x3_sub1 = self.BN(input=L3x3_sub1, training=self.training, name='inceptionB_L3x3_sub1_BN')
            L3x3_sub1 = self.parametric_relu(L3x3_sub1, 'inceptionB_L3x3_sub1_R')

            W3x3_sub2 = tf.get_variable(name='W3x3_sub2', shape=[1, n, 50, 60], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub2 = tf.nn.conv2d(name='L3x3_sub2', input=L3x3_sub1, filter=W3x3_sub2, strides=[1, 1, 2, 1], padding='SAME')
            L3x3_sub2 = self.BN(input=L3x3_sub2, training=self.training, name='inceptionB_L3x3_sub2_BN')
            L3x3_sub2 = self.parametric_relu(L3x3_sub2, 'inceptionB_L3x3_sub2_R')

            W3x3_sub3 = tf.get_variable(name='W3x3_sub3', shape=[n, 1, 60, 70], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub3 = tf.nn.conv2d(name='L3x3_sub3', input=L3x3_sub2, filter=W3x3_sub3, strides=[1, 2, 1, 1], padding='SAME')
            L3x3_sub3 = self.BN(input=L3x3_sub3, training=self.training, name='inceptionB_L3x3_sub3_BN')
            L3x3_sub3 = self.parametric_relu(L3x3_sub3, 'inceptionB_L3x3_sub3_R')

            # max pooling -> max pooling, 1x1
            L_pool = tf.nn.avg_pool(name='L_pool', value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            W_pool_sub1 = tf.get_variable(name='W_pool_sub1', shape=[1, 1, C, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L_pool_sub1 = tf.nn.conv2d(name='L_pool_sub1', input=L_pool, filter=W_pool_sub1, strides=[1, 2, 2, 1], padding='SAME')
            L_pool_sub1 = self.BN(input=L_pool_sub1, training=self.training, name='inceptionB_L_pool_sub1_BN')
            L_pool_sub1 = self.parametric_relu(L_pool_sub1, 'inceptionB_L_pool_sub1_R')

            tot_layers = tf.concat([L1x1, L5x5_sub5, L3x3_sub3, L_pool_sub1], axis=3)  # Concat in the 4th dim to stack
            self.inception_reg_losses[1] = tf.reduce_sum([tf.nn.l2_loss(train_var) for train_var in tf.get_variable_scope().trainable_variables() if re.search('W', train_var.name) is not None])
        return tot_layers

    def inception_C(self, x, n, output, name):
        B, H, W, C = x.get_shape()

        with tf.variable_scope(name):
            # 1x1
            W1x1 = tf.get_variable(name='W1x1', shape=[1, 1, C, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L1x1 = tf.nn.conv2d(name='L1x1', input=x, filter=W1x1, strides=[1, 2, 2, 1], padding='SAME')
            L1x1 = self.BN(input=L1x1, training=self.training, name='inceptionC_L1x1_BN')
            L1x1 = self.parametric_relu(L1x1, 'inceptionC_L1x1_R')

            # 5x5 -> 1x1, 1x3, 3x1, 1x3, 3x1
            W5x5_sub1 = tf.get_variable(name='W5x5_sub1', shape=[1, 1, C, 50], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub1 = tf.nn.conv2d(name='L5x5_sub1', input=x, filter=W5x5_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub1 = self.BN(input=L5x5_sub1, training=self.training, name='inceptionC_L5x5_sub1_BN')
            L5x5_sub1 = self.parametric_relu(L5x5_sub1, 'inceptionC_L5x5_sub1_R')

            W5x5_sub2 = tf.get_variable(name='W5x5_sub2', shape=[n, n, 50, 120], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub2 = tf.nn.conv2d(name='L5x5_sub2', input=L5x5_sub1, filter=W5x5_sub2, strides=[1, 2, 2, 1], padding='SAME')
            L5x5_sub2 = self.BN(input=L5x5_sub2, training=self.training, name='inceptionC_L5x5_sub2_BN')
            L5x5_sub2 = self.parametric_relu(L5x5_sub2, 'inceptionC_L5x5_sub2_R')

            W5x5_sub3 = tf.get_variable(name='W5x5_sub3', shape=[n, 1, 120, 60], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub3 = tf.nn.conv2d(name='L5x5_sub3', input=L5x5_sub2, filter=W5x5_sub3, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub3 = self.BN(input=L5x5_sub3, training=self.training, name='inceptionC_L5x5_sub3_BN')
            L5x5_sub3 = self.parametric_relu(L5x5_sub3, 'inceptionC_L5x5_sub3_R')

            W5x5_sub4 = tf.get_variable(name='W5x5_sub4', shape=[1, n, 120, 60], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub4 = tf.nn.conv2d(name='L5x5_sub4', input=L5x5_sub2, filter=W5x5_sub4, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub4 = self.BN(input=L5x5_sub4, training=self.training, name='inceptionC_L5x5_sub4_BN')
            L5x5_sub4 = self.parametric_relu(L5x5_sub4, 'inceptionC_L5x5_sub4_R')

            # 3x3 -> 1x1, 1x3, 3x1
            W3x3_sub1 = tf.get_variable(name='W3x3_sub1', shape=[1, 1, C, 120], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub1 = tf.nn.conv2d(name='L3x3_sub1', input=x, filter=W3x3_sub1, strides=[1, 2, 2, 1], padding='SAME')
            L3x3_sub1 = self.BN(input=L3x3_sub1, training=self.training, name='inceptionC_L3x3_sub1_BN')
            L3x3_sub1 = self.parametric_relu(L3x3_sub1, 'inceptionC_L3x3_sub1_R')

            W3x3_sub2 = tf.get_variable(name='W3x3_sub2', shape=[1, n, 120, 60], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub2 = tf.nn.conv2d(name='L3x3_sub2', input=L3x3_sub1, filter=W3x3_sub2, strides=[1, 1, 1, 1], padding='SAME')
            L3x3_sub2 = self.BN(input=L3x3_sub2, training=self.training, name='inceptionC_L3x3_sub2_BN')
            L3x3_sub2 = self.parametric_relu(L3x3_sub2, 'inceptionC_L3x3_sub2_R')

            W3x3_sub3 = tf.get_variable(name='W3x3_sub3', shape=[n, 1, 120, 60], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub3 = tf.nn.conv2d(name='L3x3_sub3', input=L3x3_sub1, filter=W3x3_sub3, strides=[1, 1, 1, 1], padding='SAME')
            L3x3_sub3 = self.BN(input=L3x3_sub3, training=self.training, name='inceptionC_L3x3_sub3_BN')
            L3x3_sub3 = self.parametric_relu(L3x3_sub3, 'inceptionB_L3x3_sub1_R')

            # max pooling -> max pooling, 1x1
            L_pool = tf.nn.avg_pool(name='L_pool', value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            W_pool_sub1 = tf.get_variable(name='W_pool_sub1', shape=[1, 1, C, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L_pool_sub1 = tf.nn.conv2d(name='L_pool_sub1', input=L_pool, filter=W_pool_sub1, strides=[1, 2, 2, 1], padding='SAME')
            L_pool_sub1 = self.BN(input=L_pool_sub1, training=self.training, name='inceptionC_L_pool_sub1_BN')
            L_pool_sub1 = self.parametric_relu(L_pool_sub1, 'inceptionC_L_pool_sub1_R')

            tot_layers = tf.concat([L1x1, L5x5_sub3, L5x5_sub4, L3x3_sub2, L3x3_sub3, L_pool_sub1], axis=3)  # Concat in the 4th dim to stack
            self.inception_reg_losses[2] = tf.reduce_sum([tf.nn.l2_loss(train_var) for train_var in tf.get_variable_scope().trainable_variables() if re.search('W', train_var.name) is not None])
        return tot_layers