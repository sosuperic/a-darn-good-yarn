# Main function to train and test

import argparse
import os

from core.utils.utils import combine_cmdline_and_yaml, read_yaml, setup_gpus

from network import Network
# Load and parse config

if __name__ == '__main__':
    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='main function to train and test image-sent models')

    parser.add_argument('-m', '--mode', dest='mode', default='train', help='train,test')
    parser.add_argument('-ds', '--dataset', dest='dataset', default='Sentibank', help='Sentibank,MVSO,you_imemo')
    parser.add_argument('--min_bc_cs', dest='min_bc_class_size', type=int, default=None,
                        help='when obj is bc, only use biconcepts if there is at least min_bc_cs images')
    parser.add_argument('--sent_neutral_absval', dest='sent_neutral_absval', type=float, default=None,
                        help='defines ranges [-val,val] for neutral. Images are ignored in this range for sent_biclass,\
                             while images in this range are used as neutral class for sent_triclass obj')
    parser.add_argument('--gpus', dest='gpus', default=None, help='gpu_ids to use')

    # Basic params for which job (architecture, classification goal) we're running
    # This corresponds to the training parameters set in config.yaml
    parser.add_argument('-a', '--arch', dest='arch', default='basic_cnn',
                        help='what architecture to use: basic_cnn,vgg,vgg_finetune,attention')
    parser.add_argument('-obj', dest='obj', default='sent',
                        help='What to predict: sent_reg,sent_biclass,sent_triclass,emo,bc')

    # General training params
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=None, help='batch size')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=None, help='max number of epochs')

    # Job specific training params
    parser.add_argument('-lr', dest='lr', type=float, default=None, help='learning rate')
    parser.add_argument('-optim', dest='optim', default=None, help='optimziation method')

    # Testing options - not found in yaml
    parser.add_argument('-vd', dest='video_dir', help='directory that contains video')
    parser.add_argument('-p', dest='pred', action='store_true', default=True, help='produce predictions per frame')
    parser.add_argument('-cb', dest='color_blocks', action='store_true', default=False,
                        help='predict on solid color blocks')
    parser.add_argument('-att', dest='attention', action='store_true', default=False,
                        help='produce output imgs? of where attention is focused')
    parser.add_argument('-dd', dest='deepdream', action='store_true', default=False,
                        help='produce deep dream hallucinations of filters')

    cmdline = parser.parse_args()

    # Read config from yaml
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    __cwd__ = os.path.realpath(os.getcwd())
    config = read_yaml(os.path.join(__location__, 'config.yaml'))

    # Combine command line arguments and config
    params = combine_cmdline_and_yaml(cmdline, config)
    import pprint
    pprint.pprint(params)

    # Set up GPUs
    setup_gpus(params['gpus'])

    # Get network and train/test
    network = Network(params)
    if params['mode'] == 'train':
        network.train()


