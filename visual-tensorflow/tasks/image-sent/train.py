# Train

import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

from core.utils import read_yaml
from core import basic_cnn
from core.vgg.vgg16 import vgg16
from core.vgg.imagenet_classes import class_names

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__cwd__ = os.path.realpath(os.getcwd())

config = read_yaml(os.path.join(__location__, 'config.yaml'))

# Setting up inputs
def get_files_labels_list():
    files_list = [os.path.join(__cwd__, 'data/Sentibank/aa_backup', f) for f in
             os.listdir(os.path.join(__cwd__, 'data/Sentibank/aa_backup')) if f.endswith('jpg')]
    labels_list = [1 for f in files_list]

    return files_list, labels_list

def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    img = tf.image.decode_jpeg(file_contents, channels=3)
    img.set_shape([config['model']['img_size']['h'], config['model']['img_size']['w'], 3])
    return img, label

def input_pipeline(files_tensor, labels_tensor):
    input_queue = tf.train.slice_input_producer([files_tensor, labels_tensor],
                                                shuffle=False,
                                                capacity=32
                                                )
    img, label = read_images_from_disk(input_queue)
    return img, label

def preprocess_image(image):
    image = tf.image.resize_images(image, config['model']['img_crop_size']['h'], config['model']['img_crop_size']['w'])
    image = tf.cast(image, tf.float32)
    return image

# Main setting up of graph
files_list, labels_list = get_files_labels_list()
files_tensor = tf.convert_to_tensor(files_list, dtype=tf.string)
labels_tensor = tf.convert_to_tensor(labels_list, dtype=tf.int32)
img, label = input_pipeline(files_tensor, labels_tensor)
img = preprocess_image(img)
img_batch, label_batch = tf.train.batch([img, label], batch_size=config['model']['batch_size'])

# Running
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # print img.eval()

    vgg = vgg16(img_batch, sess=sess)
    probs, label_batch = sess.run([vgg.probs, label_batch])
    print len(probs)
    for i, prob in enumerate(probs):
        print i
        preds = (np.argsort(prob)[::-1])[0:3]
        for p in preds:
            print p, class_names[p], prob[p]

    coord.request_stop()
    coord.join(threads)
