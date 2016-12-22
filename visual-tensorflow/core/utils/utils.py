# Utilities

import json
import os
from time import gmtime, strftime
import tensorflow as tf
import yaml

########################################################################################################################
# Set up, boilerplate, etc.
########################################################################################################################
def read_yaml(path):
    """Return dict from parsed yaml"""
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

    # possible_general_overwrites = ['batch_size', 'epochs']
    # for param in possible_general_overwrites:
    #     if cmdline_dict[param] is None:
    #         cmdline_dict[param] = yaml[param]

    # Job-specific, i.e. params for the given architecture + objective, etc.
    possible_job_overwrites = ['lr', 'optim']
    arch = cmdline_dict['arch']
    obj = cmdline_dict['obj']
    for param in possible_job_overwrites:
        if cmdline_dict[param] is None:
            cmdline_dict[param] = yaml[arch][obj][param]

    # Add all key-values that aren't in cmdline_dict (or are None)
    def add_remaining_kvs(yaml):
        for k,v in yaml.items():
            if isinstance(v, dict):
                add_remaining_kvs(v)
            else:
                if k in cmdline_dict:
                    if cmdline_dict[k] is None:
                        cmdline_dict[k] = v
                elif k not in cmdline_dict:
                    cmdline_dict[k] = v
    add_remaining_kvs(yaml)

    return cmdline_dict

def _convert_gpuids_to_nvidiasmi(cmdline_gpus):
    """Return inverse mapping - setting device 3 shows up as gpu 0 in nvidia-smi"""
    if cmdline_gpus is None:
        return ''
    else:
        gpus = [int(id) for id in cmdline_gpus.split(',')]
        gpus = [str(3-id) for id in gpus]
        gpus = ','.join(gpus)
        return gpus

def setup_gpus(cmdline_gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = _convert_gpuids_to_nvidiasmi(cmdline_gpus)

def make_checkpoint_dir(checkpoints_dir, params):
    """Make checkpoint dir with timestamp as name, save params as json"""
    cur_time_str = strftime("%Y-%m-%d___%H-%M-%S", gmtime())
    save_dir = os.path.join(checkpoints_dir, cur_time_str)
    os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(params, f)

    return save_dir

########################################################################################################################
# Neural networks
########################################################################################################################
def get_optimizer(optim_str, lr):
    """Return tf optimizer"""
    # optim_str = config['model']['optim']
    # lr = config['model']['lr']

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

def save_model(sess, saver, params, i):
    """Save model potentially"""
    if (i+1) % params['save_every_epoch'] == 0:
        out_file = saver.save(sess,
                              os.path.join(params['save_dir'], _get_ckpt_basename(params)),
                              global_step=i)
        print 'Model saved in file: {}'.format(out_file)

def load_model(sess, params):
    """Load model from checkpoint"""
    # First load graph
    # .meta file defines graph - they should all be the same? So just take any one
    meta_file = [f for f in os.listdir(params['ckpt_dirpath']) if f.endswith('meta')][0]
    meta_filepath = os.path.join(params['ckpt_dirpath'], meta_file)
    saver = tf.train.import_meta_graph(meta_filepath)

    # Load weights
    if params['load_epoch'] is not None:        # load the checkpoint for the given epoch
        fn = _get_ckpt_basename(params) + '-' + params['load_epoch']
        saver.restore(sess, os.path.join(params['ckpt_dirpath'], fn))
    else:       # 'checkpoint' binary file keeps track of latest checkpoints. Use it to to get most recent
        ckpt = tf.train.get_checkpoint_state(params['ckpt_dirpath'])
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(params['ckpt_dirpath'], ckpt_name)
        saver.restore(sess, ckpt_path)

    return saver

def _get_ckpt_basename(params):
    """Use to save and load"""
    return params['arch'] + '-' + params['obj']