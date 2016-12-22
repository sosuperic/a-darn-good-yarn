# Main function to train and test

import argparse
import os

from core.utils.utils import read_yaml, combine_cmdline_and_yaml

from network import Network
# Load and parse config

if __name__ == '__main__':
    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='main function to train and test image-sent models')

    parser.add_argument('-m', '--mode', dest='mode', default='train', help='train,test')
    parser.add_argument('-ds', '--dataset', dest='dataset', default='Sentibank', help='Sentibank,MVSO,you_imemo')
    parser.add_argument('--min_bc_cs', dest='min_bc_class_size', type=int, default=None,
                        help='when obj is bc, only use biconcepts if there is at least min_bc_cs images')

    # Basic params for which job (architecture, classification goal) we're running
    # This corresponds to the training parameters set in config.yaml
    parser.add_argument('-a', '--arch', dest='arch', default='basic_cnn',
                        help='what architecture to use: basic_cnn,vgg,vgg_finetune,attention')
    parser.add_argument('--load_weights', dest='load_weights', action='store_true', default=False,
                        help='load pre-trained vgg weights')
    parser.add_argument('-obj', dest='obj', default='sent',
                        help='What to predict: sent_class,emo,bc')

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

    network = Network(params)
    if params['mode'] == 'train':
        network.train()


