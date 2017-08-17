import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer, batch_norm

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input') as scope:
                self.X = tf.placeholder(dtype=tf.float32, shape=[None,32*32], name='x_data')
                self.Y = tf.placeholder(dtype=tf.float32, shape=[None,10], name='y_data')
                self.dropout_rate = tf.Variable(tf.constant(0.5),name='dropout_rate')
                self.training = tf.placeholder(dtype=tf.bool, name='training')
                X_img = tf.reshape(self.X, shape=[-1, 32 ,32 ,1])

            with tf.name_scope('stem_layer') as scope:
                self.W1_stem1 = tf.get_variable(name='W1_stem1',shape=[1, 3, 1, 40], dtype=tf.float32, initializer=variance_scaling_initializer())
                self.L1_stem1 = tf.nn.conv2d(input=X_img, filter=self.W1_stem1, strides=[1, 1, 1, 1], padding='VALID') # 32x32 -> 32x30
                self.L1_stem1 = self.BN(input=self.L1_stem1, scale=True, decay=0.99, training=self.training, name='BN_stem1')
                self.L1_stem1 = tf.nn.relu(self.L1_stem1,name='R_stem1')
                self.W1_stem2 = tf.get_variable(name='W1_stem2', shape=[3, 1, 40, 40], dtype=tf.float32, initializer=variance_scaling_initializer())
                self.L1_stem2 = tf.nn.conv2d(input=self.L1_stem1, filter=self.W1_stem2, strides=[1, 1, 1, 1], padding='VALID') # 32x30 -> 30x30
                self.L1_stem2 = self.BN(input=self.L1_stem2, scale=True, decay=0.99, training=self.training,name='BN_stem2')
                self.L1_stem2 = tf.nn.relu(self.L1_stem2, name='R_stem2')

                self.W1_stem3 = tf.get_variable(name='W1_stem3', shape=[1, 3, 40, 80], dtype=tf.float32, initializer=variance_scaling_initializer())
                self.L1_stem3 = tf.nn.conv2d(input=self.L1_stem2, filter=self.W1_stem3, strides=[1, 1, 1, 1], padding='VALID') # 30x30 -> 30x28
                self.L1_stem3 = self.BN(input=self.L1_stem3, scale=True, decay=0.99, training=self.training,name='BN_stem3')
                self.L1_stem3 = tf.nn.relu(self.L1_stem3, name='R_stem3')

                self.W1_stem4 = tf.get_variable(name='W1_stem4', shape=[3, 1, 80, 80], dtype=tf.float32, initializer=variance_scaling_initializer())
                self.L1_stem4 = tf.nn.conv2d(input=self.L1_stem3, filter=self.W1_stem4, strides=[1, 1, 1, 1],padding='VALID')  # 30x28 -> 28x2
                self.L1_stem4 = self.BN(input=self.L1_stem4, scale=True, decay=0.99, training=self.training,name='BN_stem4')
                self.L1_stem4 = tf.nn.relu(self.L1_stem4, name='R_stem4')
                self.L1_stem4 = tf.layers.dropout(inputs=self.L1_stem4, rate=self.dropout_rate, training=self.training, name='dropout_L1_stem4')

                self.W1_stem5 = tf.get_variable(name='W1_stem5', shape=[3, 3, 80, 160], dtype=tf.float32, initializer=variance_scaling_initializer())
                self.L1_stem5 = tf.nn.conv2d(input=self.L1_stem4, filter=self.W1_stem5, strides=[1, 1, 1, 1],padding='SAME')  # 28x28 -> 28x28
                self.L1_stem5 = self.BN(input=self.L1_stem5, scale=True, decay=0.99, training=self.training,name='BN_stem5')
                self.L1_stem5 = tf.nn.relu(self.L1_stem5, name='R_stem5')
                # self.L1_stem5 = tf.nn.lrn(input=self.L1_stem5, depth_radius=5, bias=2, alpha=1e-3, beta= 0.75, name='lrn_L1_stem5')

                self.L1_pool1 = tf.nn.max_pool(value=self.L1_stem5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')  # 28x28 -> 14x14
                # self.L1 = tf.nn.relu(self.L1_pool1, name='R_L1')

            with tf.name_scope('inception_layer1') as scope:
                self.L2 = self.inception_V3(input= self.L1_pool1, n=3, output=320, name= 'inception_layer1') # 14x14 -> 7x7
                self.L2 = self.BN(input= self.L2, scale=True, decay= 0.99, training=self.training, name='BN_inception1')
                # self.L2 = tf.nn.lrn(input=self.L2, depth_radius=5, bias=2, alpha=1e-3, beta=0.75, name='lrn_inception1')
                self.L2 = tf.layers.dropout(inputs=self.L2, rate=self.dropout_rate, training=self.training, name='dropout_L2')

            with tf.name_scope('inception_layer2') as scope:
                self.L3 = self.inception_V3(input=self.L2, n=3, output=640, name='inception_layer2') # 7x7 -> 4x4
                self.L3 = self.BN(input= self.L3, scale=True, decay=0.99, training=self.training, name='BN_inception2')
                # self.L3 = tf.nn.lrn(input=self.L3, depth_radius=5, bias=2, alpha=1e-3, beta=0.75, name='lrn_inception2')
                self.L3 = tf.layers.dropout(inputs=self.L3, rate=self.dropout_rate, training=self.training, name='dropout_L3')

            with tf.name_scope('conv1_layer') as scope:
                self.W4 = tf.get_variable(name='W4', shape=[3, 3, 640, 1000], dtype=tf.float32, initializer=variance_scaling_initializer())
                self.L4 = tf.nn.conv2d(input=self.L3, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME') # 4x4 -> 4x4
                self.L4 = self.BN(input= self.L4, scale=True, decay=0.99, training=self.training, name='BN_conv1')
                self.L4 = tf.nn.relu(self.L4,name='L4_R')
                # self.L4 = tf.nn.lrn(input=self.L4, depth_radius=5, bias=2, alpha=1e-3, beta=0.75)
                self.L4_pool2 = tf.nn.avg_pool(value=self.L4, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME') # 4x4 -> 1x1
                self.L4 = tf.layers.dropout(inputs=self.L4_pool2, rate=self.dropout_rate, training=self.training, name='dropout')
                self.L4 = tf.reshape(self.L4,shape=[-1, 1*1*1000])


            with tf.name_scope('fn_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[1*1*1000, 1000], dtype=tf.float32, initializer=variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc1'))
                self.L_fc1 = tf.matmul(self.L4, self.W_fc1) + self.b_fc1
                # self.L_fc1 = self.BN(input=self.L_fc1, scale=True, decay=0.99, training=self.training, name='fc1_BN')
                self.L_fc1 = tf.nn.relu(self.L_fc1, 'R_fc1')
                self.L_fc1 = tf.layers.dropout(inputs=self.L_fc1, rate=self.dropout_rate, training=self.training)

            with tf.name_scope('fn_layer2') as scope:
                self.W_fc2 = tf.get_variable(name='W_fc2', shape=[1000, 1000], dtype=tf.float32, initializer=variance_scaling_initializer())
                self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc2'))
                self.L_fc2 = tf.matmul(self.L_fc1, self.W_fc2) + self.b_fc2
                # self.L_fc2 = self.BN(input=self.L_fc2, scale=True, decay=0.99, training=self.training, name='fc2_BN')
                self.L_fc2 = tf.nn.relu(self.L_fc1, 'R_fc2')
                self.L_fc2 = tf.layers.dropout(inputs=self.L_fc1, rate=self.dropout_rate, training=self.training)

            self.W_out = tf.get_variable(name='W_out', shape=[1000, 10], dtype=tf.float32, initializer=variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[10], name='b_out'))
            self.logits = tf.matmul(self.L_fc2, self.W_out) + self.b_out

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + \
                        (0.01 / (2 * tf.to_float(tf.shape(self.Y)[0]))) * tf.reduce_sum(tf.square(self.W_out))
                        # L2 Regularization

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.dynamic_learning(0.005,early_stop_count,epoch), epsilon=1e-4).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})


    def BN(self, input, training, name, scale=True, decay=0.99):
        if training is True:
            bn = batch_norm(input, decay, scale=scale, is_training=True, updates_collections=None, scope=name)
        else:
            bn = batch_norm(input, decay, scale=scale, is_training=True, updates_collections=None, scope=name)
        return bn

    # def parametric_relu(self, _x, name):
    #     alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
    #     pos = tf.nn.relu(_x)
    #     neg = alphas * (_x - abs(_x)) * 0.5
    #     return pos + neg

    def dynamic_learning(self,learning_rate,earlystop,epoch):
        max_learning_rate = learning_rate
        min_learing_rate = 0.001
        learning_decay = 60 # 낮을수록 빨리 떨어진다.
        if earlystop >= 1:
            lr = min_learing_rate + (max_learning_rate - min_learing_rate) * np.exp(-epoch / learning_decay)
        else:
            lr = max_learning_rate
        return round(lr,4)


    def inception_V3(self, input, name, n=3, output=0):
        OPL = int(output / 4)
        B, H, W, C = input.get_shape()

        with tf.variable_scope(name):
            # 1x1
            W1x1 = tf.get_variable(name='W1x1', shape=[1, 1, C, OPL], dtype=tf.float32,initializer=variance_scaling_initializer())
            L1x1 = tf.nn.conv2d(name='L1x1', input=input, filter=W1x1, strides=[1, 2, 2, 1], padding='SAME')
            # L1x1 = self.BN(input=L1x1, scale=True, decay=0.99, training=self.training, name='inceptionV3_L1x1_BN')
            L1x1 = tf.nn.relu(L1x1, 'inceptionV3_L1x1_R')

            # -> 1x1, 3x3, 1x3, 3x1
            # 1x1
            W1x1_sub1 = tf.get_variable(name='W1x1_sub1', shape=[1, 1, C, 15], dtype=tf.float32,initializer=variance_scaling_initializer())
            L1x1_sub1 = tf.nn.conv2d(name='L1x1_sub1', input=input, filter=W1x1_sub1, strides=[1, 1, 1, 1], padding='SAME')
            # L1x1_sub1 = self.BN(input=L1x1_sub1, scale=True, decay=0.99, training=self.training, name='inceptionV3_L1x1_sub1_BN')
            L1x1_sub1 = tf.nn.relu(L1x1_sub1, 'inceptionV3_L1x1_sub1_R')
            # 3x3
            W3x3_sub1 = tf.get_variable(name='W3x3_sub1', shape=[n, n, 15, 30], dtype=tf.float32,initializer=variance_scaling_initializer())
            L3x3_sub1 = tf.nn.conv2d(name='L3x3_sub1', input=L1x1_sub1, filter=W3x3_sub1, strides=[1, 1, 1, 1],padding='SAME')
            # L1x1_sub2 = self.BN(input=L1x1_sub2, scale=True, decay=0.99, training=self.training, name='inceptionV3_L1x1_sub1_BN')
            L3x3_sub1 = tf.nn.relu(L3x3_sub1, 'inceptionV3_L3x3_sub1_R')
            # 1x3
            W1x3_sub1 = tf.get_variable(name='W1x3_sub1', shape=[1, n, 30, OPL/2], dtype=tf.float32,initializer=variance_scaling_initializer())
            L1x3_sub1 = tf.nn.conv2d(name='L1x3_sub1', input=L3x3_sub1, filter=W1x3_sub1, strides=[1, 2, 2, 1],padding='SAME')
            # L1x3_sub3 = self.BN(input=L1x3_sub3, scale=True, decay=0.99, training=self.training, name='inceptionV3_L1x1_sub1_BN')
            L1x3_sub1 = tf.nn.relu(L1x3_sub1, 'inceptionV3_L1x3_sub1_R')
            # 3x1
            W3x1_sub1 = tf.get_variable(name='W3x1_sub1', shape=[n, 1, 30, OPL/2], dtype=tf.float32,initializer=variance_scaling_initializer())
            L3x1_sub1 = tf.nn.conv2d(name='L3x1_sub1', input=L3x3_sub1, filter=W3x1_sub1, strides=[1, 2, 2, 1],padding='SAME')
            # L3x1_sub3 = self.BN(input=L3x1_sub3, scale=True, decay=0.99, training=self.training,name='inceptionV3_L3x1_sub1_BN')
            L3x1_sub1 = tf.nn.relu(L3x1_sub1, 'inceptionV3_L3x1_sub1_R')

            # -> 1x1, 1x3, 3x1
            # 1x1
            W1x1_sub2 = tf.get_variable(name='W1x1_sub2', shape=[1, 1, C, 15], dtype=tf.float32, initializer=variance_scaling_initializer())
            L1x1_sub2 = tf.nn.conv2d(name='L1x1_sub2', input=input, filter=W1x1_sub2, strides=[1, 1, 1, 1], padding='SAME')
            # L1x1_sub2 = self.BN(input=L1x1_sub2, scale=True, decay=0.99, training=self.training, name='inceptionV3_L1x1_sub2_BN')
            L1x1_sub2 = tf.nn.relu(L1x1_sub2, 'inceptionV3_L1x1_sub2_R')
            # 1x3
            W1x3_sub2 = tf.get_variable(name='W1x3_sub2', shape=[1, n, 15, OPL/2], dtype=tf.float32,initializer=variance_scaling_initializer())
            L1x3_sub2 = tf.nn.conv2d(name='L1x3_sub2', input=L1x1_sub2, filter=W1x3_sub2, strides=[1, 2, 2, 1],padding='SAME')
            # L1x3_sub3 = self.BN(input=L1x3_sub2, scale=True, decay=0.99, training=self.training, name='inceptionV3_L1x3_sub2_BN')
            L1x3_sub2 = tf.nn.relu(L1x3_sub2, 'inceptionV3_L1x3_sub2_R')
            # 3x1
            W3x1_sub2 = tf.get_variable(name='W3x1_sub2', shape=[n, 1, 15, OPL/2], dtype=tf.float32,initializer=variance_scaling_initializer())
            L3x1_sub2 = tf.nn.conv2d(name='L3x1_sub2', input=L1x1_sub2, filter=W3x1_sub2, strides=[1, 2, 2, 1],padding='SAME')
            # L3x1_sub2 = self.BN(input=L3x1_sub2, scale=True, decay=0.99, training=self.training,name='inceptionV3_L3x1_sub2_BN')
            L3x1_sub2 = tf.nn.relu(L3x1_sub2, 'inceptionV3_L3x1_sub2_R')

            # maxpool, 1x1
            L_pool = tf.nn.max_pool(name='L_pool', value=input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            W_pool_sub1 = tf.get_variable(name='W_pool_sub1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=variance_scaling_initializer())
            L_pool_sub1 = tf.nn.conv2d(name='L_pool_sub1', input=L_pool, filter=W_pool_sub1, strides=[1, 2, 2, 1], padding='SAME')
            # L_pool_sub1 = self.BN(input=L_pool_sub1, scale=True, decay=0.99, training=self.training,name='inceptionV3_L_pool_sub1_BN')
            L_pool_sub1 = tf.nn.relu(L_pool_sub1, 'inceptionV3_L_pool_sub1_R')

            tot_layers = tf.concat([L1x1, L1x3_sub1, L3x1_sub1, L1x3_sub2, L3x1_sub2, L_pool_sub1], axis=3, name='concat1')  # Concat in the 4th dim to stack
        return tot_layers

    def inception_V2(self, input, name, output, n=3):
        OPL = int(output / 4)
        B, H, W, C = input.get_shape()

        with tf.variable_scope(name):
            # 1x1
            W2_1x1 = tf.get_variable(name='W2_11x1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=variance_scaling_initializer())
            L2_1x1 = tf.nn.conv2d(name='L2_1x1', input=input, filter=W2_1x1, strides=[1, 1, 1, 1], padding='SAME')
            L2_1x1 = tf.nn.relu(L2_1x1, name='L2_R')

            # 1x1, 3x3, 3x3
            W2_1x1_sub1 = tf.get_variable(name='W2_1x1_sub1', shape=[1, 1, C, 20], dtype=tf.float32, initializer=variance_scaling_initializer())
            L2_1x1_sub1 = tf.nn.conv2d(name='L2_1x1_sub1', input=input, filter=W2_1x1_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L2_1x1_sub1 = tf.nn.relu(L2_1x1_sub1, name='L2_1x1_sub1_R')
            W2_3x3_sub1 = tf.get_variable(name='W2_3z3_sub1', shape=[n, n, 20, 40], dtype=tf.float32, initializer=variance_scaling_initializer())
            L2_3x3_sub1 = tf.nn.conv2d(name='L2_3x3_sub1', input=L2_1x1_sub1, filter=W2_3x3_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L2_3x3_sub1 = tf.nn.relu(L2_3x3_sub1, name='L2_3x3_sub1_R')
            W2_3x3_sub2 = tf.get_variable(name='W2_3x3_sub2', shape=[n, n, 40, OPL], dtype=tf.float32, initializer=variance_scaling_initializer())
            L2_3x3_sub2 = tf.nn.conv2d(name='L2_3x3_sub2', input=L2_3x3_sub1, filter=W2_3x3_sub2, strides=[1, 1, 1, 1], padding='SAME')
            L2_3x3_sub2 = tf.nn.relu(L2_3x3_sub2, name='L2_3x3_sub2_R')

            # 1x1, 3x3
            W2_1x1_sub2 = tf.get_variable(name='W2_1x1_sub2', shape=[1, 1, C, 40], dtype=tf.float32, initializer=variance_scaling_initializer())
            L2_1x1_sub2 = tf.nn.conv2d(name='L2_1x1_sub3', input=input, filter=W2_1x1_sub2, strides=[1, 1, 1, 1], padding='SAME')
            L2_1x1_sub2 = tf.nn.relu(L2_1x1_sub2, name='L2_1x1_sub2_R')
            W2_3x3_sub3 = tf.get_variable(name='W2_3x3_sub3', shape=[n, n, 40, OPL], dtype=tf.float32, initializer=variance_scaling_initializer())
            L2_3x3_sub3 = tf.nn.conv2d(name='L2_3x3_sub3', input= L2_1x1_sub2, filter=W2_3x3_sub3, strides=[1, 1, 1, 1], padding='SAME')
            L2_3x3_sub3 = tf.nn.relu(L2_3x3_sub3, name='L2_3x3_sub3_R')

            # maxpool, 1x1
            L2_maxpool = tf.nn.max_pool(name='L2_maxpool', value=input, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1,], padding='SAME')
            W2_1x1_sub3 = tf.get_variable(name='W2_1x1_sub3', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=variance_scaling_initializer())
            L2_1x1_sub3 = tf.nn.conv2d(name='L2_1x1_sub3', input=L2_maxpool, filter=W2_1x1_sub3, strides=[1, 1, 1, 1], padding='SAME')

            totlayer= tf.concat([L2_1x1, L2_3x3_sub2, L2_3x3_sub3, L2_1x1_sub3], axis=3, name='concat2')
        return totlayer

##################################################### MAIN #############################################################
import numpy as np
import time

# training_epochs = 20
batch_size = 100

train_file_list = ['data/train_data_' + str(i) + '.csv' for i in range(1, 51)]
test_file_list = ['data/test_data_' + str(i) + '.csv' for i in range(1, 11)]

def data_setting(data):
    # x : 데이터, y : 라벨
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y_tmp = np.zeros([len(data), 10])
    for i in range(0, len(data)):
        label = int(data[i][-1])
        y_tmp[i, label - 1] = 1
    y = y_tmp.tolist()

    return x, y


def read_data(filename):
    ####################################################################################################################
    ## ▣ Data Loading
    ##  - 각각의 파일에 대해 load 후 전처리를 수행
    ####################################################################################################################
    data = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(data)
    return data_setting(data)


################################################################################################################
## ▣ Data Agumentation - Created by 조원태
##  - 원본 이미지를 여러형태(90도회전, 상하반전, 좌우반전 등)로 전처리하여 이미지 데이터를 늘리는 기능
################################################################################################################
def augment_batch(train_x_batch, train_y_batch):
    rot90_list = []
    flipud_list = []
    fliplr_list = []
    batch_size = len(train_y_batch)
    for idx in range(batch_size):
        rot90_list.append(np.rot90(np.asanyarray(train_x_batch[idx]).reshape(32, 32), 1).reshape(1, 32*32))
        fliplr_list.append(np.fliplr(np.asanyarray(train_x_batch[idx]).reshape(32, 32)).reshape(1, 32*32))
        flipud_list.append(np.flipud(np.asanyarray(train_x_batch[idx]).reshape(32, 32)).reshape(1, 32*32))

    fliplr_list = np.asanyarray(fliplr_list).reshape(batch_size, 32*32)
    flipud_list = np.asanyarray(flipud_list).reshape(batch_size, 32*32)
    rot90_list = np.asanyarray(rot90_list).reshape(batch_size, 32*32)
    temp_batch_x1,temp_batch_y1 = np.append(train_x_batch, rot90_list, axis=0), np.append(train_y_batch, train_y_batch, axis=0)
    temp_batch_x2,temp_batch_y2 = np.append(fliplr_list, flipud_list, axis=0) , np.append(train_y_batch, train_y_batch, axis=0)
    temp_batch = np.c_[np.append(temp_batch_x1,temp_batch_x2, axis=0) , np.append(temp_batch_y1, temp_batch_y2, axis=0)]
    np.random.shuffle(temp_batch)
    return temp_batch[:, 0:-10].tolist(), temp_batch[:, -10:].tolist()

########################################################################################################################
## ▣ Data Training
##  - train data : 50,000 개 (10클래스, 클래스별 5,000개)
##  - epoch : 20, batch_size : 100, model : 5개
########################################################################################################################
early_stop_count = 0
epoch = 0
with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()
    models = []
    num_models = 5
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('Learning Started!')

    early_stopping_list = []
    last_epoch = -1
    # epoch = 0
    # early_stop_count = 0

    while True:
        sstime = time.time()
        avg_cost_list = np.zeros(len(models))
        for index in range(0, len(train_file_list)):
            total_x, total_y = read_data(train_file_list[index])
            # agu_x, agu_y = augment_batch(total_x, total_y)
            for start_idx in range(0, 1000, batch_size):
                train_x_batch, train_y_batch = total_x[start_idx:start_idx+batch_size], total_y[start_idx:start_idx+batch_size]

                for idx, m in enumerate(models):
                    c, _ = m.train(train_x_batch, train_y_batch)
                    avg_cost_list[idx] += c / batch_size

        ################################################################################################################
        ## ▣ early stopping - Created by 배준호
        ##  - prev epoch 과 curr epoch 의 cost 를 비교해서 curr epoch 의 cost 가 더 큰 경우 종료하는 기능
        ################################################################################################################
        saver.save(sess, 'log/epoch_' + str(epoch + 1) +'.ckpt')
        early_stopping_list.append(avg_cost_list)
        diff = 0
        if len(early_stopping_list) >= 2:
            temp = np.array(early_stopping_list)
            last_epoch = epoch
            diff = np.sum(temp[0] < temp[1])
            if diff > 2:
                early_stop_count += 1
                print('----------------------||   Early Stopped   ||----------------------')
                print('Epoch: ', '%04d' % (epoch + 1), 'cost =', avg_cost_list, ' - ', diff)
                print('early stopping - epoch({})'.format(epoch + 1), ' early stopped ', early_stop_count)
                print('---------------------------------------------------------------------')
                if early_stop_count == 3:
                   break
            early_stopping_list.pop(0)
        epoch += 1
        eetime = time.time()
        print('Epoch: ', '%04d' % (epoch), 'cost =', avg_cost_list, ' - ', diff, ', epoch{} time'.format(epoch), round(eetime - sstime, 2))

    print('Learning Finished!')

    # 종료 시간 체크
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))

tf.reset_default_graph()

########################################################################################################################
## ▣ Data Test
##  - test data : 10,000 개
##  - batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    models = []
    num_models = 5
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'log/epoch_' + str(epoch) + '.ckpt')

    print('Testing Started!')

    ensemble_accuracy = 0.
    model_accuracy = [0., 0., 0., 0., 0.]
    cnt = 0
    ensemble_confusion_mat = np.zeros((10, 10))

    for index in range(0, len(test_file_list)):
        total_x, total_y = read_data(test_file_list[index])
        for start_idx in range(0, 1000, batch_size):
            test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
            test_size = len(test_y_batch)
            predictions = np.zeros(test_size * 10).reshape(test_size, 10)

            model_result = np.zeros(test_size*10, dtype=np.int).reshape(test_size, 10)
            model_result[:, 0] = range(0, test_size)

            for idx, m in enumerate(models):
                model_accuracy[idx] += m.get_accuracy(test_x_batch, test_y_batch)
                p = m.predict(test_x_batch)
                model_result[:, 1] = np.argmax(p, 1)
                for result in model_result:
                    predictions[result[0], result[1]] += 1

            ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
            ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
            ensemble_confusion_mat = tf.add(tf.contrib.metrics.confusion_matrix(labels=tf.argmax(test_y_batch, 1),
                                                                                predictions=tf.argmax(predictions, 1),
                                                                                num_classes=10, dtype='int32',
                                                                                name='confusion_matrix'),ensemble_confusion_mat)
            cnt += 1
    for i in range(len(model_accuracy)):
        print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
    print('Testing Finished!')
    print('####### Confusion Matrix #######')
    print(sess.run(ensemble_confusion_mat))