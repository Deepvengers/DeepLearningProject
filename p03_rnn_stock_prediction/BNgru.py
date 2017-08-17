import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.contrib.layers import batch_norm, variance_scaling_initializer

class BNGRUCell(RNNCell):
    '''Batch normalized GRU as described '''

    def __init__(self, num_units, training):
        self.num_units = num_units
        self.training = training

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh', [x_size, 2 * self.num_units], initializer=variance_scaling_initializer())
            W_sh = tf.get_variable('W_hh', [self.num_units, 2 * self.num_units], initializer=variance_scaling_initializer())
            W_ixh = tf.get_variable('W_ixh', [x_size, self.num_units], initializer=variance_scaling_initializer())
            W_ish = tf.get_variable('W_ish', [self.num_units, self.num_units], initializer=variance_scaling_initializer())
            xs_bias = tf.get_variable('xs_bias', [2 * self.num_units], initializer=tf.constant_initializer(1.0))
            ixs_bias = tf.get_variable('ixs_bias', [self.num_units], initializer=tf.constant_initializer(1.0))


            xh = tf.matmul(x, W_xh)
            sh = tf.matmul(state, W_sh)
            bn_xh = bn_rnn(xh, 'xh', self.training)
            bn_sh = bn_rnn(sh, 'sh', self.training)
            hidden = bn_xh + bn_sh + xs_bias

            ixh = tf.matmul(x, W_ixh)
            ish = tf.matmul(state, W_ish)
            bn_ixh = bn_rnn(ixh, 'ixh', self.training)
            bn_ish = bn_rnn(ish, 'ish', self.training)

            u, r = tf.split(hidden, 2, 1)

            r_ = tf.nn.sigmoid(r) * bn_ish
            #todo 논문일 경우
            bn_h_ = tf.nn.softsign(bn_rnn(r_ + bn_ixh + ixs_bias, 'r_', self.training))
            new_h = tf.nn.sigmoid(u) * state + (1 - tf.nn.sigmoid(u)) * bn_h_
            #todo 논문이 아닐 경우
            # bn_h_ = tf.nn.softsign(bn_rnn(r_ + bn_ixh, 'r_', self.training))
            # new_h = tf.nn.sigmoid(u) * bn_h_ + (1 - tf.nn.sigmoid(u)) * state
            #todo 테스트 결과 논문 vs 논문x ---> 비슷비슷함.

            return new_h, new_h

def bn_rnn(x, name_scope, training, epsilon=1e-3, decay=0.999):
    '''Assume 2d [batch, values] tensor'''
    with tf.variable_scope(name_scope):
        return batch_norm(inputs=x, scale=True, epsilon=epsilon, decay=decay, updates_collections=None, is_training=training)

