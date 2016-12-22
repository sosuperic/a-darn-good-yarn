# Basic CNN
# Used for image sent

import numpy as np
import tensorflow as tf

class BasicVizsentCNN(object):
    # Following architecture of https://arxiv.org/pdf/1509.06041v1.pdf
    def __init__(self, batch_size=None, img_w=None, img_h=None, output_dim=None):
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.output_dim = output_dim

        # Create graph
        # Input
        self.img_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_h, self.img_w, 3], name='img_batch')

        # Rest of graph
        with tf.variable_scope('convpool1'):
            self.conv1 = self.conv(self.img_batch, [11, 11, 3, 96], [96], [1, 4, 4, 1])
            self.pool1 =  tf.nn.max_pool(self.conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            # print self.pool1.get_shape()
            # TODO: Normalization?

        with tf.variable_scope('convpool2'):
            self.conv2 = self.conv(self.pool1, [5, 5, 96, 256], [256], [1, 2, 2, 1])
            self.pool2 =  tf.nn.max_pool(self.conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            # TODO: Normalization?

        with tf.variable_scope('fc') as scope:
            self.reshaped = tf.reshape(self.pool2, [self.batch_size, -1])     # (B,  _)
            self.fc1 = self.fc(self.reshaped, 512, '1')         # (B, 512)
            self.fc2 = self.fc(self.fc1, 512, '2')              # (B, 512)
            self.fc3 = self.fc(self.fc2, 24, '3')               # (B, 24)
            self.fc4 = self.fc(self.fc3, self.output_dim, '4')  # (B, output_dim)

        self.last_fc = self.fc4                                 # for regression
        self.probs = tf.nn.softmax(self.fc4)                    # for classification

    def conv(self, x, kernel_shape, bias_shape, strides):
        weights = tf.get_variable('weights', kernel_shape, initializer=tf.random_normal_initializer())
        biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer())
        conv = tf.nn.conv2d(x, weights, strides, 'SAME')
        return tf.nn.relu(conv + biases)

    def get_shape(self, matrix):
        if type(matrix) == np.ndarray:
            shape = input.shape
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

        output = tf.nn.relu(tf.matmul(x, w) + b)
        if squeeze:
            output = tf.squeeze(output)

        return output