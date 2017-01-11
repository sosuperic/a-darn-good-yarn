# Utilities

import json
import logging.config
import os
import random
from time import gmtime, strftime
import tensorflow as tf
from tensorflow.python.framework import graph_util
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
    possible_job_overwrites = ['lr', 'optim', 'ckpt_dir']
    arch = cmdline_dict['arch']
    obj = cmdline_dict['obj']
    for param in possible_job_overwrites:
        if cmdline_dict[param] is None:
            if param in yaml[arch][obj]:
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

def setup_logging(default_level=logging.INFO,
                  env_key='LOG_CFG',
                  save_path=None,
                  name=None):
    """
    Setup logging configuration

    Parameters
    ----------
    default_level: default logging level
    env_key:
    save_path: filepath to save to (e.g. tasks/image-sent/logs/train.log)
    name:

    Notes
    -----
    If save_path is None, then it assumes there is a logs folder in the same directory as the file being executed
    """
    __cwd__ = os.path.realpath(os.getcwd())
    config_path = os.path.join(__cwd__, 'core/utils/logging.json')

    path = config_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)

        if os.path.isdir(save_path):
            config['handlers']['info_file_handler']['filename'] = os.path.join(save_path, 'info.log')
            config['handlers']['error_file_handler']['filename'] = os.path.join(save_path, 'error.log')
        else:
            config['handlers']['info_file_handler']['filename'] = save_path
            config['handlers']['error_file_handler']['filename'] = save_path

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    logger = logging.getLogger(name=__name__ if name is None else name)

    return logging, logger

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
        optim = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1.0)
    if optim_str == 'rmsprop':
        optim = tf.train.RMSPropOptimizer(lr)

    return optim

def save_model(sess, saver, params, i, logger):
    """Save model potentially"""
    if (i+1) % params['save_every_epoch'] == 0:
        out_file = saver.save(sess,
                              os.path.join(params['ckpt_dirpath'], _get_ckpt_basename(params)),
                              global_step=i)
        logger.info('Model saved in file: {}'.format(out_file))

def load_model(sess, params):
    """Load model from checkpoint"""
    # First load graph
    # .meta file defines graph - they should all be the same? So just take any one
    # meta_file = [f for f in os.listdir(params['ckpt_dirpath']) if f.endswith('meta')][0]
    # meta_filepath = os.path.join(params['ckpt_dirpath'], meta_file)
    # saver = tf.train.import_meta_graph(meta_filepath)

    # Load weights
    saver = tf.train.Saver(tf.global_variables())
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

def load_model_for_freeze_graph(sess, params):
    """freeze_graph is a work in progress, so haven't fully refactored load_model yet"""
    # First load graph
    # .meta file defines graph - they should all be the same? So just take any one
    # meta_file = [f for f in os.listdir(params['ckpt_dirpath']) if f.endswith('meta')][0]
    # meta_filepath = os.path.join(params['ckpt_dirpath'], meta_file)
    # saver = tf.train.import_meta_graph(meta_filepath)

    # Load weights
    saver = tf.train.Saver(tf.global_variables())
    if params['load_epoch'] is not None:        # load the checkpoint for the given epoch
        ckpt_name = _get_ckpt_basename(params) + '-' + str(params['load_epoch'])
        ckpt_path = os.path.join(params['ckpt_dirpath'], ckpt_name)
        # saver.restore(sess, os.path.join(params['ckpt_dirpath'], ckpt_name))
    else:       # 'checkpoint' binary file keeps track of latest checkpoints. Use it to to get most recent
        ckpt = tf.train.get_checkpoint_state(params['ckpt_dirpath'])
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(params['ckpt_dirpath'], ckpt_name)
    saver.restore(sess, ckpt_path)

    return saver, ckpt_path

def freeze_graph(ckpt_dirpath, arch, obj, load_epoch=None):
    params = {'ckpt_dirpath': ckpt_dirpath, 'arch': arch, 'obj': obj, 'load_epoch': load_epoch}

    # We retrieve our checkpoint fullpath
    # checkpoint = tf.train.get_checkpoint_state(model_folder)
    # input_checkpoint = checkpoint.model_checkpoint_path
    #
    # # We precise the file fullname of our freezed graph
    # absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    # output_graph = absolute_model_folder + "/frozen_model.pb"
    #
    # # Before exporting our graph, we need to precise what is our output node
    # # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # # NOTE: this variables is plural, because you can have multiple output nodes
    # output_node_names = "Accuracy/predictions"
    #
    # # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    # clear_devices = True
    #
    # # We import the meta graph and retrive a Saver
    # saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)


    output_graph = os.path.join(ckpt_dirpath, "frozen_model.pb")

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver, ckpt_path = load_model_for_freeze_graph(sess, params)
        # saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            # output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

########################################################################################################################
# Other
########################################################################################################################
def scramble_img_recursively(img, min_block_size):
    """Return recursively scrambled copy of nxn numpy array

    min_block_size should be factor of n"""
    if len(img) == min_block_size:
        return img
    else:
        copy = img.copy()
        # 4 lists - each list has 2 tuples: (y_start, y_end), (x_start, x_end)
        pixelranges = [[(0,len(img)/2),(0,len(img)/2)],
                             [(len(img)/2,len(img)),(0,len(img)/2)],
                             [(0,len(img)/2), (len(img)/2,len(img))],
                             [(len(img)/2, len(img)),(len(img)/2,len(img))]]

        indices = range(4)
        random.shuffle(indices)
        for i in range(4):
            idx = indices[i]
            copy[pixelranges[i][0][0]:pixelranges[i][0][1],pixelranges[i][1][0]:pixelranges[i][1][1],:] = \
                scramble_img_recursively(img[pixelranges[idx][0][0]:pixelranges[idx][0][1],pixelranges[idx][1][0]:pixelranges[idx][1][1],:],
                                         min_block_size)
        return copy

def scramble_img(img, block_size):
    """Return scrambled copy of nxn numpy array

    block_size should be factor of n"""
    copy = img.copy()

    img_len = len(img)
    if img_len  == block_size:
        return copy

    num_blocks_onedim = (img_len / block_size)
    num_blocks =  num_blocks_onedim ** 2

    pixelranges = []  # each sublist has 2 tuples: (y_start, y_end), (x_start, x_end)
    for i in range(num_blocks):
        y_idx = i % num_blocks_onedim
        x_idx = i / num_blocks_onedim
        y_pixelrange = (y_idx * block_size, (y_idx + 1) * block_size)
        x_pixelrange = (x_idx * block_size, (x_idx + 1) * block_size)
        pixelranges.append([y_pixelrange, x_pixelrange])

    indices = range(num_blocks)
    random.shuffle(indices)
    for i in range(num_blocks):
        idx = indices[i]
        copy[pixelranges[i][0][0]:pixelranges[i][0][1],pixelranges[i][1][0]:pixelranges[i][1][1],:] = \
            img[pixelranges[idx][0][0]:pixelranges[idx][0][1],pixelranges[idx][1][0]:pixelranges[idx][1][1]]

    return copy