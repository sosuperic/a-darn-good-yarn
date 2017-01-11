# Basic CNN
# Used for image sent

import numpy as np
import tensorflow as tf

class BasicVizsentCNN(object):
    # Following architecture of https://arxiv.org/pdf/1509.06041v1.pdf
    def __init__(self, img_w=None, img_h=None, output_dim=None, imgs=None, dropout_keep=None):
        # self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.output_dim = output_dim
        self.imgs = imgs
        self.dropout_keep = dropout_keep
        self.batch_size = self.imgs.get_shape().as_list()[0]        # variable sized batch

        # Create graph
        # Input
        self.img_batch = tf.placeholder_with_default(self.imgs,
            shape=[self.batch_size, self.img_h, self.img_w, 3], name='img_batch')
            # shape=[None, self.img_h, self.img_w, 3], name='img_batch')

        # Rest of graph
        with tf.variable_scope('convrelupool1'):
            self.conv1 = self.conv(self.img_batch, [11, 11, 3, 96], [96], [1, 4, 4, 1])
            self.relu1 = tf.nn.tanh(self.conv1)
            # self.relu1 = tf.nn.relu(self.conv1)
            self.pool1 =  tf.nn.max_pool(self.relu1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            self.norm1 = tf.nn.lrn(self.pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        with tf.variable_scope('convrelupool2'):
            self.conv2 = self.conv(self.norm1, [5, 5, 96, 256], [256], [1, 2, 2, 1])
            self.relu2 = tf.nn.tanh(self.conv2)
            # self.relu2 = tf.nn.relu(self.conv2)
            self.pool2 =  tf.nn.max_pool(self.relu2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            self.norm2 = tf.nn.lrn(self.pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        with tf.variable_scope('fc'):
            self.dropout_keep = tf.constant(self.dropout_keep)
            self.reshaped = tf.reshape(self.norm2, [self.batch_size, -1])     # (B,  _)
            self.dropout0 = tf.nn.dropout(self.reshaped, self.dropout_keep)
            self.fc1 = self.fc(self.dropout0, 512, '1')             # (B, 512)
            self.relu1 = tf.nn.tanh(self.fc1)
            # self.relu1 = tf.nn.relu(self.fc1)
            self.dropout1 = tf.nn.dropout(self.relu1, self.dropout_keep)
            self.fc2 = self.fc(self.dropout1, 512, '2')              # (B, 512)
            self.relu2 = tf.nn.tanh(self.fc2)
            # self.relu2 = tf.nn.relu(self.fc2)
            self.dropout2 = tf.nn.dropout(self.relu2, self.dropout_keep)
            self.fc3 = self.fc(self.dropout2, 24, '3')               # (B, 24)
            self.relu3 = tf.nn.tanh(self.fc3)
            # self.relu3 = tf.nn.relu(self.fc3)
            self.dropout3 = tf.nn.dropout(self.relu3, self.dropout_keep)
            self.fc4 = self.fc(self.dropout3, self.output_dim, '4')  # (B, output_dim)

        with tf.variable_scope('output'):
            self.last_fc = self.fc4
            self.probs = tf.nn.softmax(self.last_fc)

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

        output = tf.matmul(x, w) + b
        if squeeze:
            output = tf.squeeze(output)

        return output