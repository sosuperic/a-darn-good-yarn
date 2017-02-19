# Create datasets

from collections import defaultdict, Counter
import json
from natsort import natsorted
import numpy as np
import os
import pickle
import tensorflow as tf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # .../tasks/
from image_sent.prepare_data import get_bc2sent, get_label

BC_PATH = {'Sentibank': 'data/Sentibank/Flickr/bi_concepts1553',
           'MVSO': 'data/MVSO/imgs'}
BC_MEANSTD_PATH = {'Sentibank': 'data/Sentibank/Flickr/bc_channelmeanstd',
                   'MVSO': 'data/MVSO/bc_channelmeanstd'}
TFRECORDS_PATH = {'Sentibank': 'data/Sentibank/Flickr/tfrecords',
                  'MVSO': 'data/MVSO/tfrecords'}

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
        self.label2count = defaultdict(int)            # used to balance dataset

        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.__cwd__ = os.path.realpath(os.getcwd())

        self.setup_obj()

    def setup_obj(self):
        """Set up objective specific fields"""
        if self.params['obj'] == 'sent_biclass':
            self.label_dtype = tf.int32
            self.output_dim = 2

    ####################################################################################################################
    # Basic getters for public
    ####################################################################################################################
    def get_num_pts(self, split):
        return self.num_pts[split]

    def get_num_batches(self, split):
        return self.num_batches[split]

    def get_output_dim(self):
        return self.output_dim

    ####################################################################################################################
    # Methods implemented / added to by specific datasets
    ####################################################################################################################
    def get_files_list(self):
        """Returns list of files to predict on for use by tf input queue"""
        pass

    def get_tfrecords_files_list(self):
        """Returns list of files and list of labels for use by tf input queue"""
        pass

    # Create pipeline, graph, train/valid/test splits for use by network
    def read_and_decode(self, input_queue):
        pass

    def preprocess_img(self, img):
        """Basic preprocessing of image - resize to architecture's expected inputsize """
        # TODO: No need to flip because taking histogram
        # TODO: don't convert image_dtype
        # TODO: don't need to mean-std standardize?
        # TODO: basically want to return rgb values between 0,255
        img_crop_w, img_crop_h = self.params['img_crop_w'], self.params['img_crop_h']
        img = tf.image.resize_image_with_crop_or_pad(img, img_crop_h, img_crop_w)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.random_flip_left_right(img)
        return img

    def input_pipeline(self, files_list, num_read_threads=5):
        pass

    def setup_graph(self):
        pass

########################################################################################################################
###
### SENTIBANK DATASET
###
########################################################################################################################
class SentibankDataset(Dataset):
    def __init__(self, params):
        super(SentibankDataset, self).__init__(params)

    def setup_obj(self):
        super(SentibankDataset, self).setup_obj()
        if 'sent' in self.params['obj']:
            self.bc_lookup = get_bc2sent(self.params['dataset'])

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
                self.num_pts['train'] = self.num_pts['valid']
                self.num_pts['train'] = self.num_pts['valid']
            else:
                files_list['train'] = self._get_tfrecords_files_list('train')
                files_list['valid'] = self._get_tfrecords_files_list('valid')
                # Get test as well so we can get label counts and weight classes
                files_list['test'] = self._get_tfrecords_files_list('test')

            # Save label2count so we can pass it using feed_dict for weighted loss
            with open(os.path.join(self.params['ckpt_dirpath'], 'label2count.json'), 'w') as f:
                sorted_label2count = {}
                for label in sorted(self.label2count):
                    sorted_label2count[label] = self.label2count[label]
                json.dump(sorted_label2count, f)

        elif self.params['mode'] == 'test':
            files_list = self._get_tfrecords_files_list('test')

        self.num_batches = {k: int(v / self.params['batch_size']) for k,v in self.num_pts.items()}

        print self.output_dim
        print self.num_pts
        print self.num_batches

        return files_list

    def _get_tfrecords_files_list(self, split_name):
        """Helper function to return list of tfrecords files for a specific split"""
        files_list = []

        bc_meanstd_path = BC_MEANSTD_PATH[self.params['dataset']]
        tfrecords_path = TFRECORDS_PATH[self.params['dataset']]
        bc_path = BC_PATH[self.params['dataset']]

        bc2mean = pickle.load(open(os.path.join(bc_meanstd_path , 'bc2channelmean.pkl'), 'r'))
        bc2std =pickle.load(open(os.path.join(bc_meanstd_path , 'bc2channelstd.pkl'), 'r'))
        mean, std = np.zeros(3), np.zeros(3)

        # Iterate through directory, extract labels from biconcept
        tfrecords_dir = os.path.join(self.__cwd__, tfrecords_path)
        split_dir = os.path.join(tfrecords_dir, split_name)
        n = 0
        num_bc_classes = 0
        self.label2bc = {}
        for f in [f for f in os.listdir(split_dir) if not f.startswith('.')]:
            bc = os.path.basename(f).split('.')[0]

            # Potentially skip this biconcept
            # Predicting sentiment (either regression or classification)
            if 'sent' in self.params['obj']:
                if bc not in self.bc_lookup:
                    continue
            # Skip neutral concepts
            if self.params['obj'] == 'sent_biclass':
                if abs(self.bc_lookup[bc]) < self.params['sent_neutral_absval']:
                    continue
            # Skip this category if label doesn't exist
            label = get_label(self.params['dataset'], bc, self.params['obj'],
                              bc_lookup=self.bc_lookup,
                              sent_neutral_absval=self.params['sent_neutral_absval'])
            if label is None:
                continue

            # Add all tfrecord
            tfrecord_path = os.path.join(tfrecords_dir, split_name, f)
            files_list.append(tfrecord_path)

            # Add counts. TODO: this is slow (not a huge deal considering it's a one-time setup), but still...
            c = 0
            for _ in tf.python_io.tf_record_iterator(tfrecord_path):
                c += 1
            self.num_pts[split_name] += c
            self.label2count[label] += c
            n += c

            # Load mean and std stats for this biconcept
            # Running average: new average = old average * (n-c)/n + sum of new value/n).
            # Where n = total count, m = count in this update
            try:    # hmm, 3 / 1553 biconcepts are missing
                cur_bc_mean, cur_bc_std = bc2mean[bc], bc2std[bc]
                mean = (mean * (n-c) / float(n)) + (cur_bc_mean * c / float(n))
                std = (std * (n-c) / float(n)) + (cur_bc_std * c / float(n))
            except:
                continue

            # break

        # Save mean and std so we can standardize data in graph
        if self.params['mode'] == 'train':
            self.mean = mean
            self.std = std

            # Pickle so we can use at test time
            with open(os.path.join(self.params['ckpt_dirpath'], 'mean.pkl'), 'w') as f:
                pickle.dump(mean, f, protocol=2)
            with open(os.path.join(self.params['ckpt_dirpath'], 'std.pkl'), 'w') as f:
                pickle.dump(std, f, protocol=2)
        elif self.params['mode'] == 'test':
            # Load mean and std
            self.mean = pickle.load(open(os.path.join(self.params['ckpt_dirpath'], 'mean.pkl'), 'r'))
            self.std = pickle.load(open(os.path.join(self.params['ckpt_dirpath'], 'std.pkl'), 'r'))

        print 'mean: {}'.format(self.mean)
        print 'std: {}'.format(self.std)

        return files_list

    ####################################################################################################################
    # Setting up pipeline
    ####################################################################################################################
    def read_and_decode(self, input_queue):
        """Read tfrecord and decode into tensors"""
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(input_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'id': tf.FixedLenFeature([], tf.string),
                'h': tf.FixedLenFeature([], tf.int64),
                'w': tf.FixedLenFeature([], tf.int64),
                'img': tf.FixedLenFeature([], tf.string),
                'sent_biclass': tf.FixedLenFeature([], tf.int64)
            })

        h = tf.cast(features['h'], tf.int32)
        w = tf.cast(features['w'], tf.int32)
        img = tf.decode_raw(features['img'], tf.uint8)
        # img = tf.image.decode_jpeg(features['img'], channels=3)
        img = tf.reshape(img, [h, w, 3])
        img.set_shape([self.params['img_h'], self.params['img_w'], 3])
        img = self.preprocess_img(img)
        label = tf.cast(features[self.params['obj']], tf.int32)
        id = features['id']
        pred = 0.0                          # placeholder

        return img, label, id, pred

    def input_pipeline(self, files_list, num_read_threads=5):
        """Create img and label tensors from string input producer queue"""
        input_queue = tf.train.string_input_producer(files_list, shuffle=True)
        # input_queue = tf.train.string_input_producer(files_list, shuffle=True, num_epochs=self.params['epochs'])

        # TODO: where do I get num_read_threads
        with tf.device('/cpu:0'):           # save gpu for matrix ops
            img_label_id_pred_list = [self.read_and_decode(input_queue) for _ in range(num_read_threads)]
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self.params['batch_size']
        img_batch, label_batch, id_batch, pred_batch = tf.train.shuffle_batch_join(
            img_label_id_pred_list, batch_size=self.params['batch_size'], capacity=capacity,
            min_after_dequeue=min_after_dequeue)

        return img_batch, label_batch, id_batch, pred_batch

    def setup_graph(self):
        """Get lists of data, convert to tensors, set up pipeline"""
        if self.params['mode'] == 'train':
            self.splits = defaultdict(dict)
            self.files_list = self.get_tfrecords_files_list()
            for name in ['train', 'valid', 'test']:
                img_batch, label_batch, id_batch, pred_batch = self.input_pipeline(self.files_list[name])
                self.splits[name]['img_batch'] = img_batch
                self.splits[name]['label_batch'] = label_batch
                self.splits[name]['id_batch'] = id_batch
            return self.splits

        elif self.params['mode'] == 'test':
            self.files_list = self.get_tfrecords_files_list()
            img_batch, label_batch, id_batch, pred_batch = self.input_pipeline(self.files_list)
            return img_batch, label_batch

    def preprocess_img(self, img):
        img = super(SentibankDataset, self).preprocess_img(img)

        # Standardize
        img = tf.sub(img, tf.cast(tf.constant(self.mean), tf.float32))
        img = tf.div(img, tf.cast(tf.constant(self.std), tf.float32))

        return img

########################################################################################################################
###
### MVSO DATASET
###
########################################################################################################################
class MVSODataset(SentibankDataset):
    def __init__(self, params):
        super(MVSODataset, self).__init__(params)
    # TODO-refactor: SentibankDataset should just be VSODataset, update get_dataset()

########################################################################################################################
###
### PREDICTION DATASET (predicting values on unseen images, i.e. movie frames)
###
########################################################################################################################
class PredictionDataset(Dataset):
    def __init__(self, params, vid_dirpath):
        super(PredictionDataset, self).__init__(params)

        self.vid_dirpath = vid_dirpath

        # Load mean and std
        self.mean = pickle.load(open(os.path.join(self.params['ckpt_dirpath'], 'mean.pkl'), 'r'))
        self.std = pickle.load(open(os.path.join(self.params['ckpt_dirpath'], 'std.pkl'), 'r'))

    # Create pipeline, graph, train/valid/test splits for use by network
    def read_and_decode(self, input_queue):
        """Decode one image"""
        file_contents = tf.read_file(input_queue[0])

        # img = tf.decode_raw(file_contents, tf.uint8)
        # img = tf.reshape(img, [self.params['img_h'], self.params['img_w'], 3])
        # img.set_shape([self.params['img_h'], self.params['img_w'], 3])

        img = tf.image.decode_jpeg(file_contents, channels=3)
        img.set_shape([self.params['img_h'], self.params['img_w'], 3])
        return img

    def input_pipeline(self, files_tensor, num_read_threads=5):
        """Create queue"""
        input_queue = tf.train.slice_input_producer([files_tensor], shuffle=False) # capacity=1000
        img = self.read_and_decode(input_queue)
        return img

    def setup_graph(self):
        """Get lists of data, convert to tensors, set up pipeline"""
        self.files_list = self.get_files_list()
        self.files_tensor = tf.convert_to_tensor(self.files_list, dtype=tf.string)

        img = self.input_pipeline(self.files_tensor)
        img = self.preprocess_img(img)
        img_batch = tf.train.batch([img], batch_size=self.params['batch_size'])
        return img_batch

    # Get files
    def get_files_list(self):
        """Return list of images to predict"""
        files_list = natsorted([f for f in os.listdir(os.path.join(self.vid_dirpath, 'frames')) if f.endswith('jpg')])
        files_list = [os.path.join(self.vid_dirpath, 'frames', f) for f in files_list]
        self.num_pts = {'predict': len(files_list)}
        self.num_batches = {'predict': int(len(files_list) / self.params['batch_size'])}

        return files_list

    def preprocess_img(self, img):
        img = super(PredictionDataset, self).preprocess_img(img)

        # Standardize
        # img = tf.sub(img, tf.cast(tf.constant(self.mean), tf.float32))
        # img = tf.div(img, tf.cast(tf.constant(self.std), tf.float32))
        return img


def get_dataset(params, vid_dirpath=None):
    if params['mode'] == 'predict':
        return PredictionDataset(params, vid_dirpath)
    elif params['dataset'] == 'Sentibank':
        return SentibankDataset(params)
    elif params['dataset'] == 'MVSO':
        return MVSODataset(params)
