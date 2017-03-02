# Create datasets

from collections import defaultdict
import itertools
import json
from natsort import natsorted
import numpy as np
import os
import pickle
import tensorflow as tf

from prepare_data import get_bc2sent, get_bc2emo, get_bc2idx, get_label

IMGS_PATH = {'Sentibank': 'data/Sentibank/Flickr/bi_concepts1553',
           'MVSO': 'data/MVSO/imgs',
           'AVA': 'data/AVA/images'}
MEANSTD_PATH = {'Sentibank': 'data/Sentibank/Flickr/bc_channelmeanstd',
                   'MVSO': 'data/MVSO/bc_channelmeanstd',
                   'AVA': 'data/AVA/n_channelmeanstd'}
TFRECORDS_PATH = {'Sentibank': 'data/Sentibank/Flickr/tfrecords',
                  'MVSO': 'data/MVSO/tfrecords',
                  'AVA': 'data/AVA/tfrecords_0.67'}
EMOLEX_PATH = 'data/emolex/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'


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
        if self.params['obj'] == 'sent_reg':
            self.label_dtype = tf.float32
            self.output_dim = 1     # num_labels isn't the best name... oh well. used to set output_dim of network
        elif self.params['obj'] == 'sent_biclass':
            self.label_dtype = tf.int32
            self.output_dim = 2
        elif self.params['obj'] == 'sent_triclass':
            self.label_dtype = tf.int32
            self.output_dim = 3
        elif self.params['obj'] == 'emo':
            self.label_dtype = tf.int32
            if self.params['dataset'] == 'Sentibank':
                self.output_dim = 8
            elif self.params['dataset'] == 'MVSO':
                self.output_dim = 20            # 20 and not 24 b/c using emo with max val -- not all emo's are present
        elif self.params['obj'] == 'bc':
            self.label_dtype = tf.int32
            if self.params['dataset'] == 'Sentibank':
                self.output_dim = 1553          # this will be lower if min_bc_cs is actually used
            elif self.params['dataset'] == 'MVSO':
                self.output_dim = 4421          # this will be lower if min_bc_cs is actually used

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
    # Base methods
    ####################################################################################################################
    def preprocess_img(self, img):
        """Basic preprocessing of image - resize to architecture's expected inputsize """
        img_crop_w, img_crop_h = self.params['img_crop_w'], self.params['img_crop_h']
        img = tf.image.resize_image_with_crop_or_pad(img, img_crop_h, img_crop_w)
        img = tf.image.convert_image_dtype(img, tf.float32)
        if self.params['mode'] == 'train':
            img = tf.image.random_flip_left_right(img)

        return img

    def get_tfrecords_files_list(self):
        """Return list of tfrecord files"""
        if self.params['mode'] == 'train' or \
                (self.params['mode'] == 'test' and self.params['save_preds_for_prog_finetune']):
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
            # Uncomment the below line and the second self.num_baches line below to try testing on training set
            # Did that to see if actually overfitting, or bug in testing
            # files_list = self._get_tfrecords_files_list('train')

        self.num_batches = {k: int(v / self.params['batch_size']) for k,v in self.num_pts.items()}
        # self.num_batches['test'] = self.num_batches['train']

        if self.params['prog_finetune']:
            # TODO: how to calculate num_batches?
            pass

        print self.num_pts
        print self.num_batches

        return files_list

    def setup_graph(self):
        """Get lists of data, convert to tensors, set up pipeline"""
        if self.params['mode'] == 'train' or \
                (self.params['mode'] == 'test' and self.params['save_preds_for_prog_finetune']):
            self.splits = defaultdict(dict)
            self.files_list = self.get_tfrecords_files_list()
            for name in ['train', 'valid', 'test']:
                img_batch, label_batch, id_batch, pred_batch = self.input_pipeline(self.files_list[name])
                # img_batch, label_batch, id_batch = self.input_pipeline(self.files_list[name])
                self.splits[name]['img_batch'] = img_batch
                self.splits[name]['label_batch'] = label_batch
                self.splits[name]['id_batch'] = id_batch
            return self.splits

        elif self.params['mode'] == 'test':
            self.files_list = self.get_tfrecords_files_list()
            img_batch, label_batch, id_batch, pred_batch = self.input_pipeline(self.files_list)
            # img_batch, label_batch, id_batch = self.input_pipeline(self.files_list)
            return img_batch, label_batch, id_batch

########################################################################################################################
###
### SENTIBANK DATASET
###
########################################################################################################################
class SentibankDataset(Dataset):
    def __init__(self, params):
        super(SentibankDataset, self).__init__(params)

        if params['prog_finetune']:
            self.id2pred = pickle.load(open(os.path.join(self.params['ckpt_dirpath'], 'sentibank_id2pred.pkl')))

    def setup_obj(self):
        super(SentibankDataset, self).setup_obj()
        if 'sent' in self.params['obj']:
            self.bc_lookup = get_bc2sent(self.params['dataset'])
        elif self.params['obj'] == 'emo':
            self.bc_lookup = get_bc2emo(self.params['dataset'])
        elif self.params['obj'] == 'bc':
            self.bc_lookup = get_bc2idx(self.params['dataset'])

            # Some bc's are filtered out. The label from prepare_data is the index of the bc.
            # However, this means this index will exceed the output_dim of the network, and the labels won't
            # make any sense. Thus, when reading the label from the tfrecord, we want to map it to a idx
            # from [0, output_dim]
            # This will also be saved.
            self.bc_labelidx2filteredidx = {}

    ####################################################################################################################
    # Getting tfrecords list
    ####################################################################################################################
    def _get_tfrecords_files_list(self, split_name):
        """Helper function to return list of tfrecords files for a specific split"""
        # TODO: clean this up, but for now testing on shuffled tfrecords for bc (0.tfrecords instead of <bc>.tfrecords)
        if self.params['obj'] == 'bc':
            files_list = []

            # Set path
            tfrecords_path = TFRECORDS_PATH[self.params['dataset']] + '_bc'
            tfrecords_dir = os.path.join(self.__cwd__, tfrecords_path)

            # Load some pre-computed things
            self.num_pts[split_name] = pickle.load(open(os.path.join(tfrecords_dir, 'split2n.pkl'), 'rb'))[split_name]
            self.mean = pickle.load(open(os.path.join(tfrecords_dir, 'mean.pkl'), 'rb'))
            self.std = pickle.load(open(os.path.join(tfrecords_dir, 'std.pkl'), 'rb'))
            self.output_dim = pickle.load(open(os.path.join(tfrecords_dir, 'num_bc_classes.pkl'), 'rb'))
            print 'output_dim: {}'.format(self.output_dim)

            # Used when reading and decoding tfrecords to get new label
            self.bc_labelidx2filteredidx = pickle.load(open(os.path.join(tfrecords_dir,
                                                                         'bc_labelidx2filteredidx.pkl'), 'rb'))
            # Save to ckpt_dir so we can map it back to the bc later
            with open(os.path.join(self.params['ckpt_dirpath'], 'bc_labelidx2filteredidx.pkl'), 'w') as f:
                pickle.dump(self.bc_labelidx2filteredidx, f, protocol=2)
            keys = tf.constant([str(orig_idx) for orig_idx in self.bc_lookup.values()])
            values = tf.constant([self.bc_labelidx2filteredidx.get(orig_idx, -1) for orig_idx in self.bc_lookup.values()], tf.int64)
            self.label_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
            self.label_table.init.run()

            # Iterate through directory, extract labels from biconcept
            split_dir = os.path.join(tfrecords_dir, split_name)
            self.label2bc = {}
            for f in [f for f in os.listdir(split_dir) if not f.startswith('.')]:
                # Add all tfrecord
                tfrecord_path = os.path.join(tfrecords_dir, split_name, f)
                files_list.append(tfrecord_path)

            return files_list

        elif self.params['obj'] == 'sent_biclass':
            files_list = []

            # Set path
            tfrecords_path = TFRECORDS_PATH[self.params['dataset']] + '_biclass'
            tfrecords_dir = os.path.join(self.__cwd__, tfrecords_path)

            # Load some pre-computed things
            self.num_pts[split_name] = pickle.load(open(os.path.join(tfrecords_dir, 'split2n.pkl'), 'rb'))[split_name]
            self.mean = pickle.load(open(os.path.join(tfrecords_dir, 'mean.pkl'), 'rb'))
            self.std = pickle.load(open(os.path.join(tfrecords_dir, 'std.pkl'), 'rb'))
            self.output_dim = 2
            print 'output_dim: {}'.format(self.output_dim)

            # Save mean and std to checkpoint dir
            with open(os.path.join(self.params['ckpt_dirpath'], 'mean.pkl'), 'wb') as f:
                pickle.dump(self.mean, f, protocol=2)
            with open(os.path.join(self.params['ckpt_dirpath'], 'std.pkl'), 'wb') as f:
                pickle.dump(self.std, f, protocol=2)

            # Iterate through directory, extract labels from biconcept
            split_dir = os.path.join(tfrecords_dir, split_name)
            self.label2bc = {}
            for f in [f for f in os.listdir(split_dir) if not f.startswith('.')]:
                # Add all tfrecord
                tfrecord_path = os.path.join(tfrecords_dir, split_name, f)
                files_list.append(tfrecord_path)

            return files_list


        else:
            files_list = []

            bc_meanstd_path = MEANSTD_PATH[self.params['dataset']]
            tfrecords_path = TFRECORDS_PATH[self.params['dataset']]
            bc_path = IMGS_PATH[self.params['dataset']]

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
                if self.params['overfit_bc']:
                    if bc not in ['fat_pig', 'amazing_sky', 'muddy_river', 'salty_sea', 'super_market', 'dead_fish',
                                  'fat_belly', 'melted_gold', 'dark_room', 'beautiful_eyes', 'insane_clown',
                                  'funny_hat', 'sleepy_baby', 'cloudy_winter', 'abandoned_house', 'sexy_blonde',
                                  'christian_bible', 'heavy_rain', 'rotten_apple', 'fresh_rose']:
                        continue

                # Potentially skip this biconcept
                # If objective is predicting bc class, skip biconcept if not enough images
                if self.params['obj'] == 'bc':
                    num_imgs = len([tmpf for tmpf in os.listdir(os.path.join(self.__cwd__, bc_path, bc)) if tmpf.endswith('jpg')])
                    if num_imgs < self.params['min_bc_class_size']:
                        continue
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

                # This will be used to set output_dim
                if self.params['obj'] == 'bc':
                    self.bc_labelidx2filteredidx[label] = num_bc_classes
                    num_bc_classes += 1

                # Load mean and std stats for this biconcept
                # Running average: new average = old average * (n-c)/n + sum of new value/n).
                # Where n = total count, m = count in this update
                try:    # hmm, 3 / 1553 biconcepts are missing
                    cur_bc_mean, cur_bc_std = bc2mean[bc], bc2std[bc]
                    mean = (mean * (n-c) / float(n)) + (cur_bc_mean * c / float(n))
                    std = (std * (n-c) / float(n)) + (cur_bc_std * c / float(n))
                except:
                    continue

            # if num_bc_classes == 10:
            #     break
            # num_bc_classes = 880
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
                print 'Loading mean and std'
                self.mean = pickle.load(open(os.path.join(self.params['ckpt_dirpath'], 'mean.pkl'), 'r'))
                self.std = pickle.load(open(os.path.join(self.params['ckpt_dirpath'], 'std.pkl'), 'r'))

            print 'mean: {}'.format(self.mean)
            print 'std: {}'.format(self.std)

            if self.params['obj'] == 'bc':
                self.output_dim = num_bc_classes
                print 'output_dim: {}'.format(self.output_dim)

                # Save so we can map it back to the bc later
                with open(os.path.join(self.params['ckpt_dirpath'], 'bc_labelidx2filteredidx.pkl'), 'w') as f:
                    pickle.dump(self.bc_labelidx2filteredidx, f, protocol=2)

                # Used when reading and decoding tfrecords to get new label
                keys = tf.constant([str(orig_idx) for orig_idx in self.bc_lookup.values()])
                values = tf.constant([self.bc_labelidx2filteredidx.get(orig_idx, -1) for orig_idx in self.bc_lookup.values()], tf.int64)
                self.label_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
                self.label_table.init.run()

            return files_list

    ####################################################################################################################
    # Setting up pipeline
    ####################################################################################################################
    def read_and_decode(self, input_queue):
        """Read tfrecord and decode into tensors"""
        reader = tf.TFRecordReader()
        # _, serialized_example = reader.read(input_queue)
        _, serialized_examples = reader.read_up_to(input_queue, self.params['batch_size'])

        if self.params['obj'] == 'bc':
            features = tf.parse_example(serialized_examples, {
                    'id': tf.FixedLenFeature([], tf.string),
                    'h': tf.FixedLenFeature([], tf.int64),
                    'w': tf.FixedLenFeature([], tf.int64),
                    'img': tf.FixedLenFeature([], tf.string),
                    'bc': tf.FixedLenFeature([], tf.int64)
                })
        elif self.params['obj'] == 'sent_biclass':
            features = tf.parse_example(serialized_examples, {
                    'id': tf.FixedLenFeature([], tf.string),
                    'h': tf.FixedLenFeature([], tf.int64),
                    'w': tf.FixedLenFeature([], tf.int64),
                    'img': tf.FixedLenFeature([], tf.string),
                    'grayscale_hist': tf.FixedLenFeature([], tf.string),
                    'color_hist': tf.FixedLenFeature([], tf.string),
                    'sent_biclass': tf.FixedLenFeature([], tf.int64)
                })
        else:
            features = tf.parse_example(serialized_examples, {
                    'id': tf.FixedLenFeature([], tf.string),
                    'h': tf.FixedLenFeature([], tf.int64),
                    'w': tf.FixedLenFeature([], tf.int64),
                    'img': tf.FixedLenFeature([], tf.string),
                    'sent_reg': tf.FixedLenFeature([], tf.float32),
                    'sent_biclass': tf.FixedLenFeature([], tf.int64),
                    'sent_triclass': tf.FixedLenFeature([], tf.int64),
                    'emo': tf.FixedLenFeature([], tf.int64),
                    'bc': tf.FixedLenFeature([], tf.int64)
                })


        # 'id': [[id1], [id2]]

        h = tf.cast(self.params['img_h'], tf.int32)
        w = tf.cast(self.params['img_w'], tf.int32)

        def decode_img(img_raw, h=h, w=w):
            img = tf.decode_raw(img_raw, tf.uint8)
            img = tf.reshape(img, [h, w, 3])
            img.set_shape([self.params['img_h'], self.params['img_w'], 3])
            img = self.preprocess_img(img)
            return img

        def decode_label(label):
            label = tf.cast(label, tf.int32)
            if self.params['obj'] == 'bc':
                label = tf.as_string(label)
                label = self.label_table.lookup(label)
                label = tf.cast(label, tf.int32)
            return label

        def decode_pred(label):
            # Just use label as a dummy
            # if self.params['prog_finetune']:
            #     pred = self.id2pred[id]
            #     pass
            return tf.cast(0.0, tf.float32)

        # TODO: decode grayscale_hist and color_hist

        imgs = tf.map_fn(decode_img, features['img'], dtype=tf.float32,
                          back_prop=False, parallel_iterations=10)
        labels = tf.map_fn(decode_label, features[self.params['obj']], dtype=tf.int32,
                          back_prop=False, parallel_iterations=10)
        ids = features['id']
        preds = tf.map_fn(decode_pred, features['id'], dtype=tf.float32,
                          back_prop=False, parallel_iterations=10)

        return imgs, labels, ids, preds

    def input_pipeline(self, files_list, num_read_threads=5):
        """Create img and label tensors from string input producer queue"""
        input_queue = tf.train.string_input_producer(files_list, shuffle=True)

        imgs_labels_ids_preds = [self.read_and_decode(input_queue) for _ in range(num_read_threads)]

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self.params['batch_size']

        img_batch, label_batch, id_batch, pred_batch = tf.train.shuffle_batch_join(
        # img_batch, label_batch, id_batch = tf.train.shuffle_batch_join(
            imgs_labels_ids_preds,
            batch_size=self.params['batch_size'], capacity=capacity,
            min_after_dequeue=min_after_dequeue, enqueue_many=True)

        if self.params['balance']:
            if self.params['obj'] == 'sent_biclass':
                # About 2:1 ratio of positives to negatives, so only take half of positives
                neg_match = tf.to_int32(tf.equal(label_batch, [0]))              # [0,1,0,0,1]
                pos_match = tf.to_int32(tf.equal(label_batch, [1]))
                num_neg = tf.reduce_sum(neg_match)
                num_pos = tf.reduce_sum(pos_match)
                neg_match = tf.nn.top_k(neg_match, k=label_batch.get_shape().as_list()[0]).indices # sort indices [1,4,0,2,3]
                pos_match = tf.nn.top_k(pos_match, k=label_batch.get_shape().as_list()[0]).indices

                # TODO: histogram slicing dimensions will be different
                neg_imgs = tf.slice(tf.gather(img_batch, neg_match), [0,0,0,0],
                                    [num_neg, self.params['img_crop_w'], self.params['img_crop_h'], 3])
                pos_imgs = tf.slice(tf.gather(img_batch, pos_match), [0,0,0,0],
                                    [num_pos / 2 + 1, self.params['img_crop_w'], self.params['img_crop_h'], 3])
                neg_labels = tf.slice(tf.gather(label_batch, neg_match), [0], [num_neg])
                pos_labels = tf.slice(tf.gather(label_batch, pos_match), [0], [num_pos / 2 + 1])
                neg_ids = tf.slice(tf.gather(id_batch, neg_match), [0], [num_neg])
                pos_ids = tf.slice(tf.gather(id_batch, pos_match), [0], [num_pos / 2 + 1])
                neg_preds = tf.slice(tf.gather(pred_batch, neg_match), [0], [num_neg])
                pos_preds = tf.slice(tf.gather(pred_batch, pos_match), [0], [num_pos / 2 + 1])

                img_batch = tf.concat(0, [neg_imgs, pos_imgs])
                label_batch = tf.concat(0, [neg_labels, pos_labels])
                id_batch = tf.concat(0, [neg_ids, pos_ids])
                pred_batch = tf.concat(0, [neg_preds, pos_preds])


        if self.params['prog_finetune']:    # use pred_batch
            pass
            # img_batch, label_batch, id_batch, pred_batch = filter...

        return img_batch, label_batch, id_batch, pred_batch

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
### AVA DATASET
###
########################################################################################################################
class AVADataset(Dataset):
    def __init__(self, params):
        super(AVADataset, self).__init__(params)

        # Load mean and std
        meanstd_path = MEANSTD_PATH[self.params['dataset']]
        self.mean = pickle.load(open(os.path.join(meanstd_path , 'channelmean.pkl'), 'r'))
        self.std = pickle.load(open(os.path.join(meanstd_path , 'channelstd.pkl'), 'r'))
        self.num_pts = pickle.load(open(os.path.join(meanstd_path , 'split2n.pkl'), 'r'))
        self.num_batches = self.num_batches = {k: int(v / self.params['batch_size']) for k,v in self.num_pts.items()}

        print 'mean: {}'.format(self.mean)
        print 'std: {}'.format(self.std)
        print 'split2n: {}'.format(self.num_pts)

    ####################################################################################################################
    # Getting tfrecords list
    ####################################################################################################################
    def _get_tfrecords_files_list(self, split_name):
        """Helper function to return list of tfrecords files for a specific split"""
        files_list = []

        # Add all tfrecords
        tfrecords_dir = os.path.join(self.__cwd__, TFRECORDS_PATH[self.params['dataset']])
        split_dirpath = os.path.join(tfrecords_dir, split_name)
        for f in [f for f in os.listdir(split_dirpath) if not f.startswith('.')]:
            tfrecord_path = os.path.join(split_dirpath, f)
            files_list.append(tfrecord_path)

        return files_list

    ####################################################################################################################
    # Setting up pipeline
    ####################################################################################################################
    def read_and_decode(self, input_queue):
        """Read tfrecord and decode into tensors"""
        reader = tf.TFRecordReader()
        # _, serialized_example = reader.read(input_queue)
        _, serialized_examples = reader.read_up_to(input_queue, self.params['batch_size'])

        features = tf.parse_example(serialized_examples, {
                'id': tf.FixedLenFeature([], tf.string),
                'h': tf.FixedLenFeature([], tf.int64),
                'w': tf.FixedLenFeature([], tf.int64),
                'img': tf.FixedLenFeature([], tf.string),
                'sent_biclass': tf.FixedLenFeature([], tf.int64)
            })
        # 'id': [[id1], [id2]]

        def decode_img(data):
            img_raw, h, w = data[0], tf.cast(data[1], tf.int32), tf.cast(data[2], tf.int32)
            img = tf.decode_raw(img_raw, tf.uint8)
            img = tf.reshape(img, tf.pack([h, w, 3]))
            img.set_shape([self.params['img_h'], self.params['img_w'], 3])
            img = self.preprocess_img(img)
            return img

        def decode_label(label):
            label = tf.cast(label, tf.int32)
            return label

        imgs = tf.map_fn(decode_img, (features['img'], features['h'], features['w']), dtype=tf.float32,
                          back_prop=False, parallel_iterations=10)
        labels = tf.map_fn(decode_label, features[self.params['obj']], dtype=tf.int32,
                          back_prop=False, parallel_iterations=10)
        ids = features['id']

        return imgs, labels, ids

    def input_pipeline(self, files_list, num_read_threads=5):
        """Create img and label tensors from string input producer queue"""
        input_queue = tf.train.string_input_producer(files_list, shuffle=True)

        imgs_labels_ids = [self.read_and_decode(input_queue) for _ in range(num_read_threads)]

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self.params['batch_size']

        img_batch, label_batch, id_batch = tf.train.shuffle_batch_join(
            imgs_labels_ids,
            batch_size=self.params['batch_size'], capacity=capacity,
            min_after_dequeue=min_after_dequeue, enqueue_many=True)

        if self.params['balance']:
            if self.params['obj'] == 'sent_biclass':
                # About 4.75:1 ratio of positives to negatives (88128 to 18560)
                neg_match = tf.to_int32(tf.equal(label_batch, [0]))              # [0,1,0,0,1]
                pos_match = tf.to_int32(tf.equal(label_batch, [1]))
                num_neg = tf.reduce_sum(neg_match)
                num_pos = tf.cast(tf.reduce_sum(pos_match), tf.float32)
                neg_match = tf.nn.top_k(neg_match, k=label_batch.get_shape().as_list()[0]).indices # sort indices [1,4,0,2,3]
                pos_match = tf.nn.top_k(pos_match, k=label_batch.get_shape().as_list()[0]).indices

                neg_imgs = tf.slice(tf.gather(img_batch, neg_match), [0,0,0,0],
                                    [num_neg, self.params['img_crop_w'], self.params['img_crop_h'], 3])
                pos_imgs = tf.slice(tf.gather(img_batch, pos_match), [0,0,0,0],
                                    [tf.cast(tf.round(num_pos / 4.75) + 1, tf.int32),
                                     self.params['img_crop_w'], self.params['img_crop_h'], 3])
                neg_labels = tf.slice(tf.gather(label_batch, neg_match), [0], [num_neg])
                pos_labels = tf.slice(tf.gather(label_batch, pos_match), [0],
                                      [tf.cast(tf.round(num_pos / 4.75) + 1, tf.int32)])
                neg_ids = tf.slice(tf.gather(id_batch, neg_match), [0], [num_neg])
                pos_ids = tf.slice(tf.gather(id_batch, pos_match), [0],
                                   [tf.cast(tf.round(num_pos / 4.75) + 1, tf.int32)])

                img_batch = tf.concat(0, [neg_imgs, pos_imgs])
                label_batch = tf.concat(0, [neg_labels, pos_labels])
                id_batch = tf.concat(0, [neg_ids, pos_ids])

        return img_batch, label_batch, id_batch, _      # _ placeholder for pred_batch (used for fine-tuning on Sentibank)

    def preprocess_img(self, img):
        img = super(AVADataset, self).preprocess_img(img)

        # Standardize
        img = tf.sub(img, tf.cast(tf.constant(self.mean), tf.float32))
        img = tf.div(img, tf.cast(tf.constant(self.std), tf.float32))

        return img

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
        if self.params['dropout_conf']:     # repeat batch_size times, e.g. [7,9,6] -> [7,7,7,7,9,9,9,9,6,6,6,6]
            self.files_list = list(itertools.chain.from_iterable(itertools.repeat(x, self.params['batch_size'])
                                                                 for x in self.files_list))
        self.files_tensor = tf.convert_to_tensor(self.files_list, dtype=tf.string)

        img = self.input_pipeline(self.files_tensor)
        img = self.preprocess_img(img)
        img_batch = tf.train.batch([img], batch_size=self.params['batch_size'])

        self.num_pts = {'predict': len(self.files_list)}
        self.num_batches = {'predict': int(len(self.files_list) / self.params['batch_size'])}
        return img_batch

    # Get files
    def get_files_list(self):
        """Return list of images to predict"""
        files_list = natsorted([f for f in os.listdir(os.path.join(self.vid_dirpath, 'frames')) if f.endswith('jpg')])
        files_list = [os.path.join(self.vid_dirpath, 'frames', f) for f in files_list]
        return files_list

    def preprocess_img(self, img):
        img = super(PredictionDataset, self).preprocess_img(img)

        # Standardize
        img = tf.sub(img, tf.cast(tf.constant(self.mean), tf.float32))
        img = tf.div(img, tf.cast(tf.constant(self.std), tf.float32))
        return img

def get_dataset(params, vid_dirpath=None):
    if params['mode'] == 'predict':
        return PredictionDataset(params, vid_dirpath)
    elif params['dataset'] == 'Sentibank':
        return SentibankDataset(params)
    elif params['dataset'] == 'MVSO':
        return MVSODataset(params)
    elif params['dataset'] == 'AVA':
        return AVADataset(params)
