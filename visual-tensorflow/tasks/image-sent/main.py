# Main function to train and test

import argparse
import os

from core.utils.utils import read_yaml, combine_cmdline_and_yaml

from network import Network
# Load and parse config

if __name__ == '__main__':
    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='main function to train and test image-sent models')

    parser.add_argument('-m', dest='mode', default='train', help='train,test')
    parser.add_argument('-ds', dest='dataset', default='Sentibank', help='Sentibank,MVSO,you_imemo')

    # Basic params for which job (architecture, classification goal) we're running
    # This corresponds to the training parameters set in config.yaml
    parser.add_argument('-a', dest='arch', default='basic_cnn',
                        help='what architecture to use: basic_cnn,vgg,vgg_finetune,attention')
    parser.add_argument('-obj', dest='obj', default='sent',
                        help='What to predict: sent,emo,bc')

    # General training params
    parser.add_argument('-bs', dest='batch_size', default=None, help='batch size')
    parser.add_argument('-e', dest='epochs', default=None, help='max number of epochs')

    # Job specific training params
    parser.add_argument('-lr', dest='lr', default=None, help='learning rate')
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

    # network = Network(params)
    # if params['mode'] == 'train':
    #     network.train()


