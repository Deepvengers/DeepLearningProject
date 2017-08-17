import tensorflow as tf

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:

                #self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')

                self.X = tf.placeholder(tf.float32, [None, 1024], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 32, 32, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10], name='y_data')


            with tf.name_scope('conv_layer1') as scope:
                # self.W1_sub = tf.get_variable(name='W1_sub', shape=[3, 3, 1, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.L1_sub = tf.nn.conv2d(input=X_img, filter=self.W1_sub, strides=[1, 1, 1, 1], padding='VALID')  # 32*32 -> 30*30
                # self.L1_sub = self.batch_norm(self.L1_sub, shape=self.L1_sub.get_shape()[-1], training=self.training, convl=True, name='BN1')
                # self.L1_sub = self.parametric_relu(self.L1_sub, 'R1_sub')
                # self.W2_sub = tf.get_variable(name='W2_sub', shape=[3, 3, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.L2_sub = tf.nn.conv2d(input=self.L1_sub, filter=self.W2_sub, strides=[1, 1, 1, 1], padding='VALID')    # 30*30 -> 28*28
                # self.L2_sub = self.batch_norm(self.L2_sub, shape=self.L2_sub.get_shape()[-1], training = self.training, convl = True, name = 'BN2')
                # self.L2_sub = self.parametric_relu(self.L2_sub, 'R2_sub')
                # self.W3_sub = tf.get_variable(name='W3_sub', shape=[3, 3, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.L3_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W3_sub, strides=[1, 1, 1, 1], padding='VALID')    # 28*28 -> 26*26
                # self.L3_sub = self.batch_norm(self.L3_sub, shape=self.L3_sub.get_shape()[-1], training = self.training, convl = True, name = 'BN3')
                # self.L3_sub = self.parametric_relu(self.L3_sub, 'R3_sub')

                # inception module
                self.L1_sub = self.inception2d_11_33_33_maxpool11(X_img)
                #self.L1 = tf.nn.lrn(self.L3_sub, depth_radius=4, bias=2, alpha=0.0001, beta=0.75, name='LRN1')
                self.L1 = tf.nn.max_pool(value=self.L1_sub, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 26*26 -> 13*13
                #self.L1 = tf.layers.dropout(inputs=self.L1, rate=self.dropout_rate, training=self.training)


            with tf.name_scope('conv_layer2') as scope:
                self.W2 = tf.get_variable(name='W2', shape=[3, 3, 80, 80], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2 = tf.nn.conv2d(input=self.L1, filter=self.W2, strides=[1, 1, 1, 1], padding='SAME')     # 13*13
                self.L2 = self.batch_norm(self.L2, shape=self.L2.get_shape()[-1], training=self.training, convl=True, name='BN4')
                self.L2 = self.parametric_relu(self.L2, 'R2')
                #self.L2 = tf.nn.lrn(self.L2, depth_radius=6, bias=2, alpha=0.0001, beta=0.75, name='LRN2') # Local Response Normalization 구현
                self.L2 = tf.nn.max_pool(value=self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 13*13 -> 7*7
                #self.L2 = tf.layers.dropout(inputs=self.L2, rate=self.dropout_rate, training=self.training)


            with tf.name_scope('conv_layer3') as scope:
                self.W3 = tf.get_variable(name='W3', shape=[3, 3, 80, 160], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3 = tf.nn.conv2d(input=self.L2, filter=self.W3, strides=[1, 1, 1, 1], padding='SAME') # 7*7
                self.L3 = self.batch_norm(self.L3, shape=self.L3.get_shape()[-1], training=self.training, convl=True, name='BN5')
                self.L3 = self.parametric_relu(self.L3, 'R3')
                self.L3 = tf.nn.max_pool(value=self.L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 7*7 -> 4*4
                #self.L3 = tf.layers.dropout(inputs=self.L3, rate=self.dropout_rate, training=self.training)


            with tf.name_scope('conv_layer4') as scope:
                self.W4 = tf.get_variable(name='W4', shape=[3, 3, 160, 160], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4 = tf.nn.conv2d(input=self.L3, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME') # 4*4 -> 4*4
                self.L4 = self.batch_norm(self.L4, shape=self.L4.get_shape()[-1], training=self.training, convl=True, name='BN6')
                self.L4 = self.parametric_relu(self.L4, 'R4')
                # self.L4 = tf.nn.max_pool(value=self.L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                #self.L4 = tf.layers.dropout(inputs=self.L4, rate=self.dropout_rate, training=self.training)
                # self.L4 = tf.reshape(self.L4, shape=[-1, 8 * 8 * 160])


            with tf.name_scope('conv_layer5') as scope:
                self.W5 = tf.get_variable(name='W5', shape=[3, 3, 160, 320], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L5 = tf.nn.conv2d(input=self.L4, filter=self.W5, strides=[1, 1, 1, 1], padding='SAME')
                self.L5 = self.batch_norm(self.L5, shape=self.L5.get_shape()[-1], training=self.training, convl=True, name='BN7')
                self.L5 = self.parametric_relu(self.L5, 'R5')
                # self.L5 = tf.nn.max_pool(value=self.L5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 4*4 -> 4*4
                #self.L5 = tf.layers.dropout(inputs=self.L5, rate=self.dropout_rate, training=self.training)
                self.L5 = tf.reshape(self.L5, shape=[-1, 4 * 4 * 320])


            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[4 * 4 * 320, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc1'))
                self.L_fc1 = self.batch_norm(self.L5, shape=self.L5.get_shape()[-1], training=self.training, convl=False,name='BN8')
                self.L_fc1 = self.parametric_relu(tf.matmul(self.L5, self.W_fc1) + self.b_fc1, 'R_fc1')
                #self.L_fc1 = tf.layers.dropout(inputs=self.L_fc1, rate=self.dropout_rate, training=self.training)

            with tf.name_scope('fc_layer2') as scope:
                self.W_fc2 = tf.get_variable(name='W_fc2', shape=[1000, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc2'))
                self.L_fc2 = self.batch_norm(self.L_fc1, shape=self.L_fc1.get_shape()[-1], training=self.training, convl=False,name='BN9')
                self.L_fc2 = self.parametric_relu(tf.matmul(self.L_fc1, self.W_fc2) + self.b_fc2, 'R_fc2')
                #self.L_fc2 = tf.layers.dropout(inputs=self.L_fc2, rate=self.dropout_rate, training=self.training)


            self.W_out = tf.get_variable(name='W_out', shape=[1000, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[10], name='b_out'))
            self.logits = tf.matmul(self.L_fc2, self.W_out) + self.b_out


        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + (0.01/(2*tf.to_float(tf.shape(self.Y)[0])))*tf.reduce_sum(tf.square(self.W_out))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.cost)
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
        beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta')
        scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='scale')
        if convl:
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        else:
            batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)),
                            lambda: (batch_mean, batch_var))
        return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=0.001, name=name)

    def inception2d_11_33_33_maxpool11(self, x):
        # bias dimension = 3*filter_count and then the extra in_channels for the avg pooling
        #bias = tf.Variable(tf.truncated_normal([3 * filter_count + in_channels], mu, sigma)),
        self.b1 = tf.Variable(tf.constant(value=0.001, shape=[80], name='b1'))

        # 1x1
        # one_filter = tf.Variable(tf.truncated_normal([1, 1, in_channels, filter_count], mu, sigma))
        # one_by_one = tf.nn.conv2d(x, one_filter, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_1 = tf.get_variable(name='W1_1', shape=[1, 1, 1, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_1 = tf.nn.conv2d(input=x, filter=self.W1_1, strides=[1, 1, 1, 1], padding='SAME')

        # 5x5 -> 1x3 3x1 1x3 3x1
        # three_filter = tf.Variable(tf.truncated_normal([3, 3, in_channels, filter_count], mu, sigma))
        # three_by_three = tf.nn.conv2d(x, three_filter, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_2_1x3_1_sub = tf.get_variable(name='W1_2_1x3_1', shape=[1, 3, 1, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_2_1_sub = tf.nn.conv2d(input=x, filter=self.W1_2_1x3_1_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_2_3x1_1_sub = tf.get_variable(name='W1_2_3x1_1', shape=[3, 1, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_2_2_sub = tf.nn.conv2d(input=self.L1_2_1_sub, filter=self.W1_2_3x1_1_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_2_1x3_2_sub = tf.get_variable(name='W1_2_1x3_2', shape=[1, 3, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_2_3_sub = tf.nn.conv2d(input=self.L1_2_2_sub, filter=self.W1_2_1x3_2_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_2_3x1_2_sub = tf.get_variable(name='W1_2_3x1_2', shape=[3, 1, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_2 = tf.nn.conv2d(input=self.L1_2_3_sub, filter=self.W1_2_3x1_2_sub, strides=[1, 1, 1, 1], padding='SAME')

        # 3x3 -> 1x3 3x1
        # five_filter = tf.Variable(tf.truncated_normal([5, 5, in_channels, filter_count], mu, sigma))
        # five_by_five = tf.nn.conv2d(x, five_filter, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_3_1x3_sub = tf.get_variable(name='W1_3_1x3', shape=[1, 3, 1, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_3_sub = tf.nn.conv2d(input=x, filter=self.W1_3_1x3_sub, strides=[1, 1, 1, 1], padding='SAME')
        self.W1_3_3x1_sub = tf.get_variable(name='W1_3_3x1', shape=[3, 1, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_3 = tf.nn.conv2d(input=self.L1_3_sub, filter=self.W1_3_3x1_sub, strides=[1, 1, 1, 1], padding='SAME')

        # max pooling
        # pooling = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.L1_4 = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.W1_4 = tf.get_variable(name='W1_4', shape=[1, 1, 1, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        self.L1_4 = tf.nn.conv2d(input=self.L1_4, filter=self.W1_4, strides=[1, 1, 1, 1], padding='SAME')

        x = tf.concat([self.L1_1, self.L1_2, self.L1_3, self.L1_4], axis=3)  # Concat in the 4th dim to stack
        return self.parametric_relu(x + self.b1, 'L1')

