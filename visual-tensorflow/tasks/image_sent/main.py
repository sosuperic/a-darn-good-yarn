# Main function to train and test

import argparse
import os
import pprint

from core.utils.utils import combine_cmdline_and_yaml, make_checkpoint_dir, read_yaml, setup_gpus
from network import Network

if __name__ == '__main__':
    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='main function to train and test image-sent models')

    parser.add_argument('-m', '--mode', dest='mode', default='train', help='train,test,predict')
    parser.add_argument('-ds', '--dataset', dest='dataset', default='Sentibank', help='Sentibank,MVSO,you_imemo')
    parser.add_argument('--min_bc_cs', dest='min_bc_class_size', type=int, default=None,
                        help='when obj is bc, only use biconcepts if there is at least min_bc_cs images')
    parser.add_argument('--sent_neutral_absval', dest='sent_neutral_absval', type=float, default=None,
                        help='defines ranges [-val,val] for neutral. Images are ignored in this range for sent_biclass,\
                             while images in this range are used as neutral class for sent_triclass obj')

    # Basic params for which job (architecture, classification goal) we're running
    # This corresponds to the training parameters set in config.yaml
    parser.add_argument('-a', '--arch', dest='arch', default='basic_cnn',
                        help='what architecture to use: basic_cnn,vgg,vgg_finetune,attention')
    parser.add_argument('-obj', dest='obj', default='sent_biclass',
                        help='What to predict: sent_reg,sent_biclass,sent_triclass,emo,bc')

    # General training params
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=None, help='batch size')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=None, help='max number of epochs')

    # Job specific training params
    parser.add_argument('-lr', dest='lr', type=float, default=None, help='learning rate')
    parser.add_argument('-optim', dest='optim', default=None, help='optimziation method')

    # Testing options - not found in yaml
    parser.add_argument('-vd', '--video_dir', dest='video_dir', help='directory that contains video and frames/ folder')
    parser.add_argument('-cb', dest='color_blocks', action='store_true', default=False,
                        help='predict on solid color blocks')
    parser.add_argument('-att', dest='attention', action='store_true', default=False,
                        help='produce output imgs? of where attention is focused')
    parser.add_argument('-dd', dest='deepdream', action='store_true', default=False,
                        help='produce deep dream hallucinations of filters')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default=None, help='directory to load checkpointed model')
    parser.add_argument('--load_epoch', dest='load_epoch', default=None, help='checkpoint epoch to load')

    # Bookkeeping, checkpointing, etc.
    parser.add_argument('--save_every_epoch', dest='save_every_epoch', type=int, default=None,
                        help='save model every _ epochs')
    parser.add_argument('--val_every_epoch', dest='val_every_epoch', type=int, default=None,
                        help='evaluate on validation set every _ epochs')
    parser.add_argument('--gpus', dest='gpus', default=None, help='gpu_ids to use')

    cmdline = parser.parse_args()

    ####################################################################################################################

    # Read config from yaml
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config = read_yaml(os.path.join(__location__, 'config.yaml'))

    # Combine command line arguments and config
    params = combine_cmdline_and_yaml(cmdline, config)

    # Set up GPUs
    setup_gpus(params['gpus'])

    # Print params
    pprint.pprint(params)

    # Train / test
    if params['mode'] == 'train':
        # Make checkpoint directory
        checkpoints_dir = os.path.join(__location__, 'checkpoints')
        save_dir = make_checkpoint_dir(checkpoints_dir, params)
        params['save_dir'] = save_dir
        print save_dir

        network = Network(params)
        network.train()

    elif params['mode'] == 'test':
        params['ckpt_dirpath'] = os.path.join(__location__, 'checkpoints', params['ckpt_dir'])
        network = Network(params)
        network.test()

    elif params['mode'] == 'predict':
        params['ckpt_dirpath'] = os.path.join(__location__, 'checkpoints', params['ckpt_dir'])
        network = Network(params)
        network.predict()


