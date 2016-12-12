# Utilities

import tensorflow as tf
import yaml

def read_yaml(path):
    """Return parsed yaml"""
    with open(path, 'r') as f:
        try:
            return yaml.load(f)
        except yaml.YAMLError as e:
            print e

def combine_cmdline_and_yaml(cmdline, yaml):
    """
    Return dict of parameter to value

    Parameters
    ----------
    cmdline: argparse object
    yaml: dictionary returned by read_yaml

    Returns
    -------
    cmdline_dict: (flat) dict of param names to values

    Notes
    -----
    Cmdline take precedence over yaml. If parameter is not defined in cmdline, pull it from yaml. yaml used to store
    values that either a) values that are more hard coded (image sizes for different architectures), or b)
    'best' parameters found after cross validation
    """
    cmdline_dict = vars(cmdline)

    possible_general_overwrites = ['batch_size', 'epochs']
    for param in possible_general_overwrites:
        if cmdline_dict[param] is None:
            cmdline_dict[param] = yaml[param]

    # Job-specific, i.e. params for given architecture, objective, etc.
    possible_job_overwrites = ['lr', 'optim']
    arch = cmdline_dict['arch']
    obj = cmdline_dict['obj']
    for param in possible_job_overwrites:
        if cmdline_dict[param] is None:
            cmdline_dict[param] = yaml[arch][obj][param]

    return cmdline_dict

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
