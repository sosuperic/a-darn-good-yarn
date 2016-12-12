# Create datasets

from collections import defaultdict
import os
import re
import tensorflow as tf

class Dataset(object):
    def __init__(self, params):
        self.params = params

        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.__cwd__ = os.path.realpath(os.getcwd())

    # Methods implemented / added to by specific datasets
    def get_files_labels_list(self):
        """Returns list of files and list of labels for use by tf input queue"""
        pass

    def get_splits(self):
        """TODO: train/valid/test"""
        pass

    def preprocess_img(self, img):
        """Basic preprocessing of image - resize to architecture's expected inputsize """
        img_crop_w, img_crop_h = self.params['img_crop_w'], self.params['img_crop_h']
        img = tf.image.resize_image_with_crop_or_pad(img, img_crop_h, img_crop_w)
        img = tf.cast(img, tf.float32)
        return img

    # Methods used by all datasets
    def get_num_pts(self):
        return self.num_pts

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
        self.labels_tensor = tf.convert_to_tensor(self.labels_list, dtype=tf.int32)
        self.img, self.label = self._input_pipeline(self.files_tensor, self.labels_tensor)
        self.img = self.preprocess_img(self.img)
        self.img_batch, self.label_batch = tf.train.batch([self.img, self.label],
                                                          batch_size=self.params['batch_size'])
        return self.img_batch, self.label_batch

    def test(self):
        return 2

class SentibankDataset(Dataset):
    def __init__(self, params):
        super(SentibankDataset, self).__init__(params)
        self.bc2sent = self.get_bc2sent()

        print self.params

    def _get_label(self, bc):
        """Return label from bi_concept string according to the objective (sentiment, emotion, biconcept)"""
        if self.params['obj'] == 'sent':
            return self.bc2sent[bc]
        if self.params['obj'] == 'bc':
            # TODO: one hot encode? Will have to know total number of classes tho (some bc's may be skipped)
            return bc

    # Override parent method
    def get_files_labels_list(self):
        files_list = []
        labels_list = []

        # Iterate through directory, extract labels from biconcept
        sentibank_img_dir = os.path.join(self.__cwd__, 'data/Sentibank/Flickr/bi_concepts1553')
        for bc_dir in [d for d in os.listdir(sentibank_img_dir) if not d.startswith('.')]:
            bc_path = os.path.join(sentibank_img_dir, bc_dir)
            bc_files_list = [os.path.join(bc_path, f) for f in os.listdir(bc_path) if f.endswith('jpg')]

            # If objective is predicting bc class, skip biconcept if not enough images
            if self.params['obj'] == 'bc':
                if len(bc_files_list) < self.params['min_bc_class_size']:
                    continue

            # Add all images
            files_list += bc_files_list

            # Add all labels
            label = self._get_label(bc_dir)
            bc_labels_list = [label] * len(bc_files_list)
            labels_list += bc_labels_list

            break
        print files_list
        print labels_list

        return files_list, labels_list

    def preprocess_img(self, img):
        img = super(SentibankDataset, self).preprocess_img(img)
        # Do more things to img
        return img

    def get_bc2sent(self):
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


def get_dataset(params):
    if params['dataset'] == 'Sentibank':
        return SentibankDataset(params)