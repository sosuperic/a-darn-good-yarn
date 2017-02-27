# Modified (very similar) AlexNet - w bn

import numpy as np
import tensorflow as tf

class ModifiedAlexNet(object):
    # Following architecture of DeepSentibank (https://arxiv.org/pdf/1410.8586v1.pdf)
    # def __init__(self, img_w=None, img_h=None, output_dim=None, imgs=None, dropout_keep=None):
    def __init__(self, img_w=None, img_h=None, output_dim=None, imgs=None, dropout_keep=None, bn_decay=None, is_training=None):
        self.img_w = img_w
        self.img_h = img_h
        self.output_dim = output_dim
        self.imgs = imgs
        self.dropout_keep = tf.constant(dropout_keep)
        self.batch_size = self.imgs.get_shape().as_list()[0]        # variable sized batch
        self.bn_decay = bn_decay
        self.is_training = is_training
        self.bn_reuse = False if self.is_training else True
        # print 'MAN {}'.format(self.is_training)

        # Input
        self.img_batch = tf.placeholder_with_default(self.imgs,
            shape=[self.batch_size, self.img_h, self.img_w, 3], name='img_batch')

        # Rest of graph
        with tf.variable_scope('conv1') as scope:
            self.conv1 = self.conv(self.img_batch, [11, 11, 3, 96], [96], [1, 4, 4, 1])
            # self.bn1 = self.conv1
            # self.bn1 = self.batch_norm_wrapper(self.conv1, self.is_training, self.bn_decay, scope, self.bn_reuse)
            # self.bn1 = tf.contrib.layers.batch_norm(self.conv1, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            self.bn1 = tf.contrib.layers.batch_norm(self.conv1, decay=self.bn_decay, is_training=self.is_training, updates_collections=None, fused=True)
            # self.relu1 = tf.nn.relu(self.bn1)
            self.relu1 = self.prelu(self.bn1, 'prelu1')
            self.pool1 =  tf.nn.max_pool(self.relu1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
            # self.norm1 = tf.nn.lrn(self.pool1, depth_radius=5, bias=2.0, alpha=0.0001, beta=0.75)
            self.norm1 = tf.nn.lrn(self.pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        with tf.variable_scope('conv2') as scope:
            # self.conv2 = self.conv(self.norm1, [5, 5, 96, 128], [128], [1, 1, 1, 1])
            self.conv2 = self.conv(self.norm1, [5, 5, 96, 256], [256], [1, 1, 1, 1])
            # self.bn2 = self.conv2
            # self.bn2 = self.batch_norm_wrapper(self.conv2, self.is_training, self.bn_decay, scope, self.bn_reuse)
            # self.bn2 = tf.contrib.layers.batch_norm(self.conv2, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            self.bn2 = tf.contrib.layers.batch_norm(self.conv2, decay=self.bn_decay, is_training=self.is_training, updates_collections=None, fused=True)
            # self.relu2 = tf.nn.relu(self.bn2)
            self.relu2 = self.prelu(self.bn2, 'prelu2')
            self.pool2 = tf.nn.max_pool(self.relu2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
            # self.norm2 = tf.nn.lrn(self.pool2, depth_radius=5, bias=2.0, alpha=0.0001, beta=0.75)
            self.norm2 = tf.nn.lrn(self.pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        with tf.variable_scope('conv3') as scope:
            # self.conv3 = self.conv(self.norm2, [3, 3, 128, 384], [384], [1, 1, 1, 1])
            self.conv3 = self.conv(self.norm2, [3, 3, 256, 384], [384], [1, 1, 1, 1])
            # self.bn3 = self.conv3
            # self.bn3 = self.batch_norm_wrapper(self.conv3, self.is_training, self.bn_decay, scope, self.bn_reuse)
            # self.bn3 = tf.contrib.layers.batch_norm(self.conv3, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            self.bn3 = tf.contrib.layers.batch_norm(self.conv3, decay=self.bn_decay, is_training=self.is_training, updates_collections=None, fused=True)
            # self.relu3 = tf.nn.relu(self.bn3)
            self.relu3 = self.prelu(self.bn3, 'prelu3')

        with tf.variable_scope('conv4') as scope:
            # self.conv4 = self.conv(self.relu3, [3, 3, 384, 192], [192], [1, 1, 1, 1])
            self.conv4 = self.conv(self.relu3, [3, 3, 384, 384], [384], [1, 1, 1, 1])
            # self.bn4 = self.conv4
            # self.bn4 = self.batch_norm_wrapper(self.conv4, self.is_training, self.bn_decay, scope, self.bn_reuse)
            # self.bn4 = tf.contrib.layers.batch_norm(self.conv4, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            self.bn4 = tf.contrib.layers.batch_norm(self.conv4, decay=self.bn_decay, is_training=self.is_training, updates_collections=None, fused=True)
            # self.relu4 = tf.nn.relu(self.bn4)
            self.relu4 = self.prelu(self.bn4, 'prelu4')

        with tf.variable_scope('conv5') as scope:
            # self.conv5 = self.conv(self.relu4, [3, 3, 192, 128], [128], [1, 1, 1, 1])
            self.conv5 = self.conv(self.relu4, [3, 3, 384, 256], [256], [1, 1, 1, 1])
            # self.bn5 = self.conv5
            # self.bn5 = self.batch_norm_wrapper(self.conv5, self.is_training, self.bn_decay, scope, self.bn_reuse)
            # self.bn5 = tf.contrib.layers.batch_norm(self.conv5, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            self.bn5 = tf.contrib.layers.batch_norm(self.conv5, decay=self.bn_decay, is_training=self.is_training, updates_collections=None, fused=True)
            # self.relu5 = tf.nn.   relu(self.bn5)
            self.relu5 = self.prelu(self.bn5, 'prelu5')
            self.pool5 = tf.nn.max_pool(self.relu5, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        with tf.variable_scope('fc6'):
            self.reshaped = tf.reshape(self.pool5, [self.batch_size, -1])
            # self.fc6 = self.fc(self.reshaped, 4096, 'fc6')
            self.fc6 = self.fc(self.reshaped, 2048, 'fc6')
            # self.fc6 = self.fc(self.reshaped, 1024, 'fc6')
            # self.relu6 = tf.nn.relu(self.fc6)
            self.relu6 = self.prelu(self.fc6, 'prelu6')
            self.dropout6 = tf.nn.dropout(self.relu6, self.dropout_keep)

        with tf.variable_scope('fc7'):
            # self.fc7 = self.fc(self.dropout6, 4096, 'fc7')
            self.fc7 = self.fc(self.dropout6, 2048, 'fc7')
            # self.fc7 = self.fc(self.dropout6, 1024, 'fc7')
            # self.relu7 = tf.nn.relu(self.fc7)
            self.relu7 = self.prelu(self.fc7, 'prelu7')
            self.dropout7 = tf.nn.dropout(self.relu7, self.dropout_keep)

        with tf.variable_scope('fc8'):
            self.fc8 = self.fc(self.dropout7, self.output_dim, 'fc8')

        with tf.variable_scope('output'):
            self.last_fc = self.fc8
            self.probs = tf.nn.softmax(self.last_fc)
            # tf.summary.histogram("last_fc", self.last_fc)
            # tf.summary.histogram("probs", self.probs)

    def conv(self, x, kernel_shape, bias_shape, strides, stddev=0.01):
        weights = tf.get_variable('weights', kernel_shape, initializer=tf.random_normal_initializer(stddev=stddev))
        # weights = tf.get_variable('weights', kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(x, weights, strides, 'SAME')
        conv += biases
        return conv

    def get_shape(self, matrix):
        if type(matrix) == np.ndarray:
            shape = matrix.shape
        else:   # tensor
            shape = matrix.get_shape().as_list()
        return shape

    def fc(self, x, output_dim, name, squeeze=False, stddev=0.01):
        """
        x: can be 1D (dim1, ) or 2D (dim1, dim2)
            can be numpy array (maybe if we're feeding in to feed_dict) or tensor
        squeeze: squeezes output. Can be useful for final FC layer to produce probabilities for each class

        Returns x*w + b of dim (dim1, output_dim)
        """
        shape = self.get_shape(x)
        if len(shape) == 1:
            input_dim = shape[0]
            x = tf.reshape(x, [1, -1])  # (dim1, ) -> (1, dim1)
        if len(shape) == 2:
            input_dim = shape[1]
        if len(shape) > 2:
            raise Exception('Shape issues')

        w_name = '{}_w'.format(name)
        b_name = '{}_b'.format(name)
        w = tf.get_variable(w_name, [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=stddev))
        # w = tf.get_variable(w_name, [input_dim, output_dim], tf.float32, tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(b_name, [output_dim], tf.float32, tf.constant_initializer(0.1))

        output = tf.matmul(x, w) + b
        if squeeze:
            output = tf.squeeze(output)

        return output

    def prelu(self, x, name):
        alphas = tf.get_variable(name, x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5

        return pos + neg

    def batch_norm_wrapper(self, x, phase, decay, scope, reuse):
        # with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope(scope):
            normed = tf.contrib.layers.batch_norm(x, center=True, scale=True, decay=decay,
                                                  is_training=phase, scope='bn', #reuse=reuse,
                                                  updates_collections=None, fused=True)
            return normed