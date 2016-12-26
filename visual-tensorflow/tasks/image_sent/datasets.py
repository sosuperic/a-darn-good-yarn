# Create datasets

from collections import defaultdict, Counter
import json
from natsort import natsorted
import os
import tensorflow as tf

from prepare_data import get_bc2sent, get_bc2emo, get_bc2idx, get_label

BC_PATH = 'data/Sentibank/Flickr/bi_concepts1553'
TFRECORDS_PATH = 'data/Sentibank/Flickr/tfrecords'
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
            self.output_dim = 8
        elif self.params['obj'] == 'bc':
            self.label_dtype = tf.int32
            self.output_dim = 1553

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
        img_crop_w, img_crop_h = self.params['img_crop_w'], self.params['img_crop_h']
        img = tf.image.resize_image_with_crop_or_pad(img, img_crop_h, img_crop_w)
        img = tf.image.convert_image_dtype(img, tf.float32)
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
            self.bc_lookup = get_bc2sent()
        elif self.params['obj'] == 'emo':
            self.bc_lookup = get_bc2emo()
        elif self.params['obj'] == 'bc':
            self.bc_lookup = get_bc2idx()

    ####################################################################################################################
    # Getting tfrecords list
    ####################################################################################################################
    def get_tfrecords_files_list(self):
        """Return list of tfrecord files"""
        if self.params['mode'] == 'train':
            files_list = {}
            # files_list['train'] = self._get_tfrecords_files_list('train')
            files_list['valid'] = self._get_tfrecords_files_list('valid')
            # Get test as well so we can get label counts and weight classes
            # files_list['test'] = self._get_tfrecords_files_list('test')

            # For debugging: uncomment this, and comment out the above train = and test =
            files_list['train'] = files_list['valid']
            files_list['test'] = files_list['valid']
            self.num_pts['train'] = self.num_pts['valid']
            self.num_pts['train'] = self.num_pts['valid']

            # Save label2count so we can pass it using feed_dict for loss
            with open(os.path.join(self.params['save_dir'], 'label2count.json'), 'w') as f:
                sorted_label2count = {}
                for label in sorted(self.label2count):
                    sorted_label2count[label] = self.label2count[label]
                json.dump(sorted_label2count, f)

        elif self.params['mode'] == 'test':
            files_list = self._get_tfrecords_files_list('test')

        self.num_batches = {k: int(v / self.params['batch_size']) for k,v in self.num_pts.items()}
        print self.num_pts
        print self.num_batches

        return files_list

    def _get_tfrecords_files_list(self, split_name):
        """Helper function to return list of tfrecords files for a specific split"""
        files_list = []

        # Iterate through directory, extract labels from biconcept
        tfrecords_dir = os.path.join(self.__cwd__, TFRECORDS_PATH)
        split_dir = os.path.join(tfrecords_dir, split_name)
        for f in [f for f in os.listdir(split_dir) if not f.startswith('.')]:
            bc = os.path.basename(f).split('.')[0]

            # Potentially skip this biconcept
            # If objective is predicting bc class, skip biconcept if not enough images
            if self.params['obj'] == 'bc':
                num_imgs = len(os.listdir(os.path.join(self.__cwd__, BC_PATH, bc)))
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
            label = get_label(bc, self.params['obj'],
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
                'h': tf.FixedLenFeature([], tf.int64),
                'w': tf.FixedLenFeature([], tf.int64),
                'img': tf.FixedLenFeature([], tf.string),
                'sent_reg': tf.FixedLenFeature([], tf.float32),
                'sent_biclass': tf.FixedLenFeature([], tf.int64),
                'sent_triclass': tf.FixedLenFeature([], tf.int64),
                'emo': tf.FixedLenFeature([], tf.int64),
                'bc': tf.FixedLenFeature([], tf.int64)
            })

        h = tf.cast(features['h'], tf.int32)
        w = tf.cast(features['w'], tf.int32)
        img = tf.decode_raw(features['img'], tf.uint8)
        # img = tf.image.decode_jpeg(features['img'], channels=3)
        img = tf.reshape(img, [h, w, 3])
        img.set_shape([self.params['img_h'], self.params['img_w'], 3])

        label = tf.cast(features[self.params['obj']], tf.int32)

        img = self.preprocess_img(img)

        return img, label

    def input_pipeline(self, files_list, num_read_threads=5):
        """Create img and label tensors from string input producer queue"""
        input_queue = tf.train.string_input_producer(files_list, shuffle=True)

        # TODO: where do I get num_read_threads
        with tf.device('/cpu:0'):       # save gpu for matrix ops
            img_label_list = [self.read_and_decode(input_queue) for _ in range(num_read_threads)]
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self.params['batch_size']
        img_batch, label_batch = tf.train.shuffle_batch_join(
            img_label_list, batch_size=self.params['batch_size'], capacity=capacity,
            min_after_dequeue=min_after_dequeue)

        return img_batch, label_batch

    def setup_graph(self):
        """Get lists of data, convert to tensors, set up pipeline"""
        if self.params['mode'] == 'train':
            self.splits = defaultdict(dict)
            self.files_list = self.get_tfrecords_files_list()
            split_names = {0: 'train', 1: 'valid', 2: 'test'}
            for i in range(3):
                name = split_names[i]
                img_batch, label_batch = self.input_pipeline(self.files_list[name])
                self.splits[name]['img_batch'] = img_batch
                self.splits[name]['label_batch'] = label_batch
            return self.splits

        elif self.params['mode'] == 'test':
            self.files_list = self.get_tfrecords_files_list()
            img_batch, label_batch = self.input_pipeline(self.files_list)
            return img_batch, label_batch

    def preprocess_img(self, img):
        img = super(SentibankDataset, self).preprocess_img(img)
        # Do more things to img
        return img

########################################################################################################################
###
### PREDICTION DATASET (predicting values on unseen images, i.e. movie frames)
###
########################################################################################################################
class PredictionDataset(Dataset):

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

        img = self.input_pipeline(self.files_list)
        img = self.preprocess_img(img)
        img_batch = tf.train.batch([img], batch_size=self.params['batch_size'])
        return img_batch

    # Get files
    def get_files_list(self):
        """Return list of images to predict"""
        files_list = natsorted([f for f in os.listdir(os.path.join(self.params['video_dir'], 'frames')) if f.endswith('jpg')])
        files_list = [os.path.join(self.params['video_dir'], 'frames', f) for f in files_list]
        self.num_pts = {'predict': len(files_list)}
        self.num_batches = {'predict': int(len(files_list) / self.params['batch_size'])}

        return files_list

def get_dataset(params):
    if params['mode'] == 'predict':
        return PredictionDataset(params)
    elif params['dataset'] == 'Sentibank':
        return SentibankDataset(params)
