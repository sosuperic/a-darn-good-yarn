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
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='only use valid set so speed up loading (dont load train')
    parser.add_argument('-ds', '--dataset', dest='dataset', default='Sentibank', help='Sentibank,MVSO,you_imemo')
    parser.add_argument('--min_bc_cs', dest='min_bc_class_size', type=int, default=None,
                        help='when obj is bc, only use biconcepts if there is at least min_bc_cs images')
    parser.add_argument('--sent_neutral_absval', dest='sent_neutral_absval', type=float, default=None,
                        help='defines ranges [-val,val] for neutral. Images are ignored in this range for sent_biclass,\
                             while images in this range are used as neutral class for sent_triclass obj')

    # Loading model for fine-tuning or testing
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='load model and finetune')
    parser.add_argument('--prog_finetune', dest='prog_finetune', action='store_true',
                        help='load model and progressively finetune using datapts that have high-confidence predictions')
    parser.add_argument('--save_preds_for_prog_finetune', dest='save_preds_for_prog_finetune', action='store_true',
                        help='save predictions to pkl file for progressively fine-tuning; used with mode=test')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default=None, help='directory to load checkpointed model')
    parser.add_argument('--load_epoch', dest='load_epoch', default=None, help='checkpoint epoch to load')

    # Basic params for which job (architecture, classification goal) we're running
    # This corresponds to the training parameters set in config.yaml
    parser.add_argument('-a', '--arch', dest='arch', default='basic_cnn',
                        help='what architecture to use: basic_cnn,vgg,vgg_finetune,attention')
    parser.add_argument('-obj', dest='obj', default='sent_biclass',
                        help='What to predict: sent_reg,sent_biclass,sent_triclass,emo,bc')

    # General training params
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=None, help='batch size')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=None, help='max number of epochs')
    parser.add_argument('--dropout', dest='dropout', type=float, default=None,
                        help='use 1.0 when testing -- tensorflow uses keep_prob')
    parser.add_argument('--weight_classes', dest='weight_classes', action='store_true', default=False,
                        help='weight classes for class imbalance')

    # Job specific training params
    parser.add_argument('-lr', dest='lr', type=float, default=None, help='learning rate')
    parser.add_argument('-optim', dest='optim', default=None, help='optimziation method')

    # Testing options - not found in yaml
    parser.add_argument('-vd', '--vid_dirpath', dest='vid_dirpath',
                        help='either (a) path to directory that contains video and frames/ folder, or '\
                             '(b) directory that contains subdirs that have video and frames/ '\
                             'used to with mode=predict')
    parser.add_argument('--scramble_img_mode', dest='scramble_img_mode', default=None,
                        help='uniform,recursive; used with mode=test')
    parser.add_argument('--scramble_blocksize', dest='scramble_blocksize', default=None, type=int,
                        help='multiple of 2 in range [2,128]')

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
        ckpt_dirpath = make_checkpoint_dir(checkpoints_dir, params)
        params['ckpt_dirpath'] = ckpt_dirpath
        print ckpt_dirpath
        network = Network(params)
        network.train()

    elif params['mode'] == 'test':
        params['dropout'] = 1.0
        params['ckpt_dirpath'] = os.path.join(__location__, 'checkpoints', params['ckpt_dir'])
        network = Network(params)
        network.test()

    elif params['mode'] == 'predict':
        params['dropout'] = 1.0
        params['ckpt_dirpath'] = os.path.join(__location__, 'checkpoints', params['ckpt_dir'])
        network = Network(params)
        network.predict()




