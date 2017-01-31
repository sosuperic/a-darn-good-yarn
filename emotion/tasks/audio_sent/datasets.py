# Create datasets

from collections import defaultdict
import cPickle as pickle
import json
import numpy as np
import os
import tensorflow as tf

from prepare_data import MELGRAM_30S_SIZE

TFRECORDS_PATH = 'data/spotify/tfrecords'
NUMPTS_AND_MEANSTD_PATH = 'data/spotify/numpts_and_meanstd.pkl'
MELGRAM_20S_SIZE = [96, 938]

#######################################################################################################################
###
### BASE DATASET CLASS
###
########################################################################################################################
class Dataset(object):
    def __init__(self, params):
        self.params = params
        self.num_pts = defaultdict(int)
        self.num_batches = {}

        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.__cwd__ = os.path.realpath(os.getcwd())

        self.setup_obj()

    def setup_obj(self):
        if self.params['obj'] == 'valence_reg':
            self.label_dtype = tf.float32
            self.output_dim = 1

    ####################################################################################################################
    # Basic getters for public
    ####################################################################################################################
    def get_num_pts(self, split):
        return self.num_pts[split]

    def get_num_batches(self, split):
        return self.num_batches[split]

    def get_output_dim(self):
        return self.output_dim

########################################################################################################################
###
### SPOTIFY DATASET
###
########################################################################################################################
class SpotifyDataset(Dataset):
    def __init__(self, params):
        super(SpotifyDataset, self).__init__(params)
        self.load_numpts_and_meanstd()
        self.num_batches = {k: int(v / self.params['batch_size']) for k,v in self.num_pts.items()}

    def load_numpts_and_meanstd(self):
        """
        Load the pre-computed number of points per train-valid-test split and per mel-bin mean and stddev so we can
        normalize data.
        """
        numpts_and_meanstd = pickle.load(open(NUMPTS_AND_MEANSTD_PATH, 'rb'))
        for split in ['train', 'valid', 'test']:
            self.num_pts[split] = numpts_and_meanstd['num_pts'][split]
        self.mean = numpts_and_meanstd['mean']      # (number of mel-bins, 1)
        self.std = numpts_and_meanstd['std']        # (number of mel-bins, 1)

        print self.num_pts
        print self.mean.shape
        print self.std.shape

    ####################################################################################################################
    # Getting tfrecords list
    ####################################################################################################################
    def get_tfrecords_files_list(self):
        """Return list of tfrecord files"""
        if self.params['mode'] == 'train':
            files_list = {}
            if self.params['debug']:
                # Only retrieve valid, set train and test to it
                files_list['valid'] = self._get_tfrecords_files_list('valid')
                files_list['train'] = files_list['valid']
                files_list['test'] = files_list['valid']
            else:
                files_list['train'] = self._get_tfrecords_files_list('train')
                files_list['valid'] = self._get_tfrecords_files_list('valid')
                # files_list['test'] = self._get_tfrecords_files_list('test')
        elif self.params['mode'] == 'test':
            files_list = self._get_tfrecords_files_list('test')

        return files_list

    def _get_tfrecords_files_list(self, split_name):
        """
        Helper function to return list of tfrecords files for a specific split
        """
        files_list = []

        # Iterate through directory, extract labels from biconcept
        tfrecords_dir = os.path.join(self.__cwd__, TFRECORDS_PATH)
        split_dir = os.path.join(tfrecords_dir, split_name)
        for f in [f for f in os.listdir(split_dir) if not f.startswith('.')]:
            tfrecord_path = os.path.join(tfrecords_dir, split_name, f)
            files_list.append(tfrecord_path)

        return files_list

    ####################################################################################################################
    # Setting up pipeline
    ####################################################################################################################
    def preprocess_feats(self, feats):
        """
        1) Select random 20-second clip from original 30-second record
        2) z-standardize per mel-bin

        feats: mel-spectrogram, 1st dimension is mel-bin, 2nd dimension is time
        """
        # Random crop
        feats = tf.random_crop(feats, MELGRAM_20S_SIZE)

        # Standardize
        feats = tf.sub(feats, tf.cast(tf.constant(self.mean), tf.float32))
        feats = tf.div(feats, tf.cast(tf.constant(self.std), tf.float32))

        return feats

    def read_and_decode(self, input_queue):
        """
        Read tfrecord and decode into tensors
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(input_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'id': tf.FixedLenFeature([], tf.string),
                'log_melgram': tf.FixedLenFeature([], tf.string),
                'valence_reg': tf.FixedLenFeature([], tf.float32)
            })

        # feats = tf.decode_raw(features['log_melgram'], tf.float64)
        feats = tf.decode_raw(features['log_melgram'], tf.float32)
        feats = tf.cast(feats, tf.float32)
        # feats = tf.decode_raw(features['log_melgram'], tf.uint8)
        feats = tf.reshape(feats, MELGRAM_30S_SIZE)
        feats.set_shape(MELGRAM_30S_SIZE)
        feats = self.preprocess_feats(feats)
        feats = tf.expand_dims(feats, 2)        # (batch, mel-bins, time) -> (batch, mel-bins, time, 1)

        label = features[self.params['obj']]

        return feats, label

    def input_pipeline(self, files_list, num_read_threads=5):
        """
        Create img and label tensors from string input producer queue
        """
        input_queue = tf.train.string_input_producer(files_list, shuffle=True)

        # TODO: where do I get num_read_threads
        with tf.device('/cpu:0'):           # save gpu for matrix ops
            clip_label_list = [self.read_and_decode(input_queue) for _ in range(num_read_threads)]
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self.params['batch_size']
        clip_batch, label_batch = tf.train.shuffle_batch_join(
            clip_label_list,
            batch_size=self.params['batch_size'],
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)

        return clip_batch, label_batch

    def setup_graph(self):
        """
        Get lists of data, convert to tensors, set up pipeline
        """
        if self.params['mode'] == 'train':
            self.splits = defaultdict(dict)
            self.files_list = self.get_tfrecords_files_list()
            for name in ['train', 'valid']:
                clip_batch, label_batch = self.input_pipeline(self.files_list[name])
                self.splits[name]['clip_batch'] = clip_batch
                self.splits[name]['label_batch'] = label_batch
            return self.splits

        elif self.params['mode'] == 'test':
            self.files_list = self.get_tfrecords_files_list()
            clip_batch, label_batch = self.input_pipeline(self.files_list)
            return clip_batch, label_batch

########################################################################################################################
###
### PREDICTION DATASET (predicting values on unseen images, i.e. movie frames)
###
########################################################################################################################
class PredictionDataset(Dataset):
    def __init__(self, params, vid_dirpath):
        pass

def get_dataset(params, vid_dirpath=None):
    if params['mode'] == 'predict':
        return PredictionDataset(params, vid_dirpath)
    else:
        return SpotifyDataset(params)
