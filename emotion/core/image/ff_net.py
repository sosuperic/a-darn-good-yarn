# 2 hidden layer feed forward net used with gray/rgb histogram features

import numpy as np
import tensorflow as tf

class FFNet(object):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, imgs=None, dropout_keep=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.imgs = imgs
        self.dropout_keep = dropout_keep
        self.batch_size = self.imgs.get_shape().as_list()[0]        # variable sized batch

        self.img_batch = tf.placeholder_with_default(self.imgs,
            shape=[self.batch_size, self.input_dim], name='img_batch')

        # Rest of graph
        with tf.variable_scope('fc'):
            self.fc1 = self.fc(self.img_batch, self.hidden_dim, '1')
            self.relu1 = tf.nn.relu(self.fc1)
            self.dropout1 = tf.nn.dropout(self.relu1, self.dropout_keep)

            self.fc2 = self.fc(self.dropout1, self.hidden_dim, '2')
            self.relu2 = tf.nn.relu(self.fc2)
            self.dropout2 = tf.nn.dropout(self.relu2, self.dropout_keep)

            self.fc3 = self.fc(self.dropout2, self.output_dim, '3')

        with tf.variable_scope('output'):
            self.last_fc = self.fc3
            self.probs = tf.nn.softmax(self.last_fc)

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