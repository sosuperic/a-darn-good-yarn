# Fully convolutional network on audio clip inputs
# Used for audio sent

import numpy as np
from datasets import MELGRAM_20S_SIZE
import tensorflow as tf

class AudioCNN(object):
    def __init__(self, clips=None, output_dim=None, bn_decay=None, is_training=None):
        self.clips = clips
        self.batch_size = self.clips.get_shape().as_list()[0]        # variable sized batch
        # print self.batch_size
        # self.batch_size = tf.shape(clips)[0]
        self.output_dim = output_dim
        self.bn_decay = bn_decay
        self.is_training = is_training

        # Create graph
        # Input
        self.clips_batch = tf.placeholder_with_default(
            self.clips,
            shape=[self.batch_size, MELGRAM_20S_SIZE[0], MELGRAM_20S_SIZE[1], 1],
            # shape=[None, MELGRAM_20S_SIZE[0], MELGRAM_20S_SIZE[1], 1],
            # shape = tf.concat(0, [self.batch_size,  [MELGRAM_20S_SIZE[0], MELGRAM_20S_SIZE[1], 1]]),
            name='clip_batch')

        # Rest of graph
        with tf.variable_scope('convblock1'):
            self.conv1 = self.conv(self.clips_batch, [3, 3, 1, 32], [32], [1, 1, 1, 1])
            self.bn1 = tf.contrib.layers.batch_norm(self.conv1, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            # self.bn1 = tf.contrib.layers.batch_norm(self.conv1, is_training=self.is_training)
            self.elu1 = tf.nn.elu(self.bn1)
            self.pool1 = tf.nn.max_pool(self.elu1, [1, 2, 4, 1], [1, 2, 4, 1], 'SAME')        # wait should ti be 1,2,4,1 for stride?

        with tf.variable_scope('convblock2'):
            self.conv2 = self.conv(self.pool1, [3, 3, 32, 128], [128], [1, 1, 1, 1])
            self.bn2 = tf.contrib.layers.batch_norm(self.conv2, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            # self.bn2 = tf.contrib.layers.batch_norm(self.conv2, is_training=self.is_training)
            self.elu2 = tf.nn.elu(self.bn2)
            self.pool2 = tf.nn.max_pool(self.elu2, [1, 2, 4, 1], [1, 2, 4, 1], 'SAME')

        with tf.variable_scope('convblock3'):
            self.conv3 = self.conv(self.pool2, [3, 3, 128, 128], [128], [1, 1, 1, 1])
            self.bn3 = tf.contrib.layers.batch_norm(self.conv3, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            # self.bn3 = tf.contrib.layers.batch_norm(self.conv3, is_training=self.is_training)
            self.elu3 = tf.nn.elu(self.bn3)
            self.pool3 = tf.nn.max_pool(self.elu3, [1, 2, 4, 1], [1, 2, 4, 1], 'SAME')

        with tf.variable_scope('convblock4'):
            self.conv4 = self.conv(self.pool3, [3, 3, 128, 192], [192], [1, 1, 1, 1])
            self.bn4 = tf.contrib.layers.batch_norm(self.conv4, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            # self.bn4 = tf.contrib.layers.batch_norm(self.conv4, is_training=self.is_training)
            self.elu4 = tf.nn.elu(self.bn4)
            self.pool4 = tf.nn.max_pool(self.elu4, [1, 2, 4, 1], [1, 2, 4, 1], 'SAME')

        with tf.variable_scope('convblock5'):
            self.conv5 = self.conv(self.pool4, [3, 3, 192, 256], [256], [1, 1, 1, 1])
            self.bn5 = tf.contrib.layers.batch_norm(self.conv5, decay=self.bn_decay, is_training=self.is_training, updates_collections=None)
            # self.bn5 = tf.contrib.layers.batch_norm(self.conv5, is_training=self.is_training)
            self.elu5 = tf.nn.elu(self.bn5)
            self.pool5 = tf.nn.max_pool(self.elu5, [1, 4, 4, 1], [1, 4, 4, 1], 'SAME')

            # print self.pool5.get_shape().as_list() # None, 2, 1, 256

        with tf.variable_scope('output'):
            # self.reshaped = tf.reshape(self.pool5, [None, -1])
            self.reshaped = tf.reshape(self.pool5, [self.batch_size, -1])
            # print self.reshaped.get_shape()
            self.fc = self.fc(self.reshaped, self.output_dim, '1')
            self.fc_sigmoid = tf.nn.sigmoid(self.fc)
            self.out = self.fc_sigmoid

    def conv(self, x, kernel_shape, bias_shape, strides):
        weights = tf.get_variable('weights', kernel_shape, initializer=tf.random_normal_initializer())
        biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer())
        conv = tf.nn.conv2d(x, weights, strides, 'SAME')
        return tf.nn.relu(conv + biases)

    def get_shape(self, matrix):
        if type(matrix) == np.ndarray:
            shape = matrix.shape
        else:   # tensor
            shape = matrix.get_shape().as_list()
        return shape

    def fc(self, x, output_dim, name, squeeze=False):
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
            raise Exception ('Shape issues')

        w_name = '{}_w'.format(name)
        b_name = '{}_b'.format(name)
        w = tf.get_variable(w_name, [input_dim, output_dim], tf.float32, tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(b_name, [output_dim], tf.float32, tf.constant_initializer(0.0))

        output = tf.matmul(x, w) + b
        if squeeze:
            output = tf.squeeze(output)

        return output