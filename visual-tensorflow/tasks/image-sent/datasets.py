# Create datasets

from collections import defaultdict, Counter
import os
import re
import tensorflow as tf

BC_PATH = 'data/Sentibank/Flickr/bi_concepts1553'
EMOLEX_PATH = 'data/emolex/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'


#######################################################################################################################
###
### BASE DATASET CLASS
###
########################################################################################################################
class Dataset(object):
    def __init__(self, params):
        self.params = params

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
    def get_num_pts(self):
        return self.num_pts

    def get_num_batches(self):
        return int(self.num_pts / self.params['batch_size'])

    def get_output_dim(self):
        return self.output_dim

    ####################################################################################################################
    # Methods implemented / added to by specific datasets
    ####################################################################################################################
    def get_files_labels_list(self):
        """Returns list of files and list of labels for use by tf input queue"""
        pass

    def preprocess_img(self, img):
        """Basic preprocessing of image - resize to architecture's expected inputsize """
        img_crop_w, img_crop_h = self.params['img_crop_w'], self.params['img_crop_h']
        img = tf.image.resize_image_with_crop_or_pad(img, img_crop_h, img_crop_w)
        img = tf.cast(img, tf.float32)
        return img

    ####################################################################################################################
    # Create pipeline, graph, train/valid/test splits for use by network
    ####################################################################################################################
    def _read_images_from_disk(self, input_queue):
        file_contents, label = tf.read_file(input_queue[0]), input_queue[1]
        img = tf.image.decode_jpeg(file_contents, channels=3)
        img.set_shape([self.params['img_h'], self.params['img_w'], 3])
        return img, label

    def _input_pipeline(self, files_tensor, labels_tensor):
        input_queue = tf.train.slice_input_producer([files_tensor, labels_tensor],
                                                    shuffle=False,
                                                    capacity=32
                                                    )
        img, label = self._read_images_from_disk(input_queue)
        return img, label

    def setup_graph(self):
        self.files_list, self.labels_list = self.get_files_labels_list()
        self.files_tensor = tf.convert_to_tensor(self.files_list, dtype=tf.string)
        self.labels_tensor = tf.convert_to_tensor(self.labels_list, dtype=self.label_dtype)
        self.img, self.label = self._input_pipeline(self.files_tensor, self.labels_tensor)
        self.img = self.preprocess_img(self.img)
        self.img_batch, self.label_batch = tf.train.batch([self.img, self.label],
                                                          batch_size=self.params['batch_size'])
        return self.img_batch, self.label_batch

    def get_splits(self):
        """TODO: train/valid/test"""
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
            self.bc2sent = self._get_bc2sent()
        elif self.params['obj'] == 'emo':
            self.bc2emo = self._get_bc2emo()
        elif self.params['obj'] == 'bc':
            self.bc2idx = self._get_bc2idx()

    ####################################################################################################################
    # Getting labels
    ####################################################################################################################
    def _map_label_to_int(self, label):
        """Map emo and bc string labels to int for classification tasks"""
        if self.params['obj'] == 'sent_biclass':
            label = 'neg' if label < 0 else 'pos'
            d = {'neg': 0, 'pos': 1}
            return d[label]
        elif self.params['obj'] == 'sent_triclass':
            if label > self.params['sent_neutral_absval']:
                label = 'pos'
            elif label < -1 * self.params['sent_neutral_absval']:
                label = 'neg'
            else:
                label = 'neutral'
            d = {'neg':0, 'neutral':1, 'pos':2}
            return d[label]
        elif self.params['obj'] == 'emo':
            d = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3,
                 'joy': 4, 'sadness': 5, 'surprise': 6, 'trust': 7}
            return d[label]
        elif self.params['obj'] == 'bc':
            return self.bc2idx[label]


    def _get_label(self, bc):
        """
        Return label from bi_concept string according to the objective (sentiment, emotion, biconcept)

        Handful of cases for sent where label doesn't exist. For example, candid_guy
        """
        if self.params['obj'] == 'sent_reg':
            if bc in self.bc2sent:
                return self.bc2sent[bc]
            else:
                None
        elif self.params['obj'] == 'sent_biclass' or self.params['obj'] == 'sent_triclass':
            if bc in self.bc2sent:
                return self._map_label_to_int(self.bc2sent[bc])
            else:
                return None
        elif self.params['obj'] == 'emo':
            if len(self.bc2emo[bc]) > 0:
                emo = max(self.bc2emo[bc])
                return self._map_label_to_int(emo)
            else:       # no emotions for biconcept
                return None
        elif self.params['obj'] == 'bc':
            return self._map_label_to_int(bc)

    ####################################################################################################################
    # Overriding / adding to parent methods
    ####################################################################################################################
    def get_files_labels_list(self):
        files_list = []
        labels_list = []

        # Iterate through directory, extract labels from biconcept
        sentibank_img_dir = os.path.join(self.__cwd__, BC_PATH)
        for bc_dir in [d for d in os.listdir(sentibank_img_dir) if not d.startswith('.')]:
            bc_path = os.path.join(sentibank_img_dir, bc_dir)
            bc_files_list = [os.path.join(bc_path, f) for f in os.listdir(bc_path) if f.endswith('jpg')]

            # Potentially skip this biconcept
            # If objective is predicting bc class, skip biconcept if not enough images
            if self.params['obj'] == 'bc':
                if len(bc_files_list) < self.params['min_bc_class_size']:
                    continue
            # Predicting sentiment (either regression or classification)
            if 'sent' in self.params['obj']:
                if bc_dir not in self.bc2sent:
                    continue
            # Skip neutral concepts
            if self.params['obj'] == 'sent_biclass':
                if abs(self.bc2sent[bc_dir]) < self.params['sent_neutral_absval']:
                    continue

            # Skip this category if label doesn't exist
            label = self._get_label(bc_dir)
            if label is None:
                continue

            # Add all images
            files_list += bc_files_list

            # Add all labels
            bc_labels_list = [label] * len(bc_files_list)
            labels_list += bc_labels_list

        self.num_pts = len(files_list)

        return files_list, labels_list

    def preprocess_img(self, img):
        img = super(SentibankDataset, self).preprocess_img(img)
        # Do more things to img
        return img

    ####################################################################################################################
    # Getting data structures for each objective
    ####################################################################################################################
    ### Function to return data structures for getting labels from biconcept
    def _get_bc2sent(self):
        """
        Return dictionary mapping bi_concept to positive-negative polarity values

        Example line: frail_hand [sentiment: -1.44] [#imgs: 358]
            - note: number of imgs is number returned by Flickr - not necessarily what's in dataset
        """
        bc2sent_and_count = {}
        with open('data/Sentibank/VSO/3244ANPs.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.endswith(']') and '_' in line:
                    m = re.match(r'(.+_.+)\s\[sentiment:\s(.+)\]\s\[#imgs:\s(.+)\]', line)
                    bc, sent, _ = m.group(1), float(m.group(2)), int(m.group(3).replace(',', ''))
                    bc2sent_and_count[bc] = sent
        return bc2sent_and_count

    def _get_bc2emo(self):
        """
        Use emolex to map bi_concept to emotions. Return dict with bc as key, counts of emotions as values.

        Stats: 857 bc's with at least one emotion (57.3%) , 696 emotions without any emotions
        """
        def get_emolex():
            word2emotions = defaultdict(set)
            f = open(EMOLEX_PATH, 'rb')
            i = 0
            for line in f.readlines():
                if i > 45:          # Previous lines are readme
                    word, emotion, flag = line.strip('\n').split()
                    if emotion == 'positive' or emotion == 'negative':
                        continue
                    if int(flag) == 1:
                        word2emotions[word].add(emotion)
                i += 1
            return word2emotions

        bc2emo = defaultdict(list)
        bc2img_fps = self._get_all_bc_img_fps()
        word2emotions = get_emolex()
        for bc, _ in bc2img_fps.items():
            adj, noun = bc.split('_')
            if adj in word2emotions:
                for emotion in word2emotions[adj]:
                    bc2emo[bc].append(emotion)
            if noun in word2emotions:
                for emotion in word2emotions[noun]:
                    bc2emo[bc].append(emotion)
            bc2emo[bc] = Counter(bc2emo[bc])

        return bc2emo

    def _get_all_bc_img_fps(self):
        """Return dictionary mapping bi_concept to list of img file paths"""
        bc2img_fps = {}
        for bc in [d for d in os.listdir(BC_PATH) if not d.startswith('.')]:
            cur_bc_path = os.path.join(BC_PATH, bc)
            img_fns = [f for f in os.listdir(cur_bc_path) if f.endswith('jpg')]
            img_fps = [os.path.join(cur_bc_path, fn) for fn in img_fns]
            bc2img_fps[bc] = img_fps

        return bc2img_fps

    def _get_bc2idx(self):
        """Return dictionary mapping biconcept to idx"""
        bc2idx = {}
        for i, bc in enumerate([d for d in os.listdir(BC_PATH) if not d.startswith('.')]):
            bc2idx[bc] = i
        return bc2idx


def get_dataset(params):
    if params['dataset'] == 'Sentibank':
        return SentibankDataset(params)