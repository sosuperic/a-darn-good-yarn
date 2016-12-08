# Utilities
# TODO: make this into a class?

import tensorflow as tf
import yaml


def read_yaml(path):
    """Return parsed yaml"""
    with open(path, 'r') as f:
        try:
            return yaml.load(f)
        except yaml.YAMLError as e:
            print e

def get_optimizer(config):
    """Return tf optimizer"""
    optim_str = config['model']['optim']
    lr = config['model']['lr']

    if optim_str == 'sgd':
        optim = tf.train.GradientDescentOptimizer(lr)
    if optim_str == 'adadelta':
        optim = tf.train.AdadeltaOptimizer(learning_rate=lr)
    if optim_str == 'adagrad':
        optim = tf.train.AdagradOptimizer(lr)
    if optim_str == 'adam':
        optim = tf.train.AdamOptimizer(learning_rate=lr)
    if optim_str == 'rmsprop':
        optim = tf.train.RMSPropOptimizer(lr)

    return optim
