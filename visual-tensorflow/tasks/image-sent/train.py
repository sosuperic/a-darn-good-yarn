# Train

import os
import tensorflow as tf
from scipy.misc import imread, imresize

from core.utils import read_yaml, get_optimizer
from core import basic_cnn
from core.vgg.vgg16 import vgg16

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__cwd__ = os.path.realpath(os.getcwd())

config = read_yaml(os.path.join(__location__, 'config.yaml'))

# Setting up inputs
def get_files_labels_list():
    files_list = [os.path.join(__cwd__, 'data/Sentibank/aa_backup', f) for f in
             os.listdir(os.path.join(__cwd__, 'data/Sentibank/aa_backup')) if f.endswith('jpg')]
    labels_list = [3 for f in files_list] # 4th class in imagenet

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
    # Finish setting up graph - vgg requires sess
    vgg = vgg16(batch_size=config['model']['batch_size'],
            w=config['model']['img_crop_size']['w'],
            h=config['model']['img_crop_size']['h'],
            sess=sess,
            load_weights=True)
    # Set up training
    num_classes = 1000
    labels_onehot = tf.one_hot(label_batch, num_classes)     # (batch_size, 1000)
    optimizer = get_optimizer(config)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(vgg.probs, labels_onehot))
    train_step = optimizer.minimize(cross_entropy)

    # Initialize after optimization - this needs to be done after adam
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Training
    for i in range(config['model']['epochs']):
        # Normally slice_input_producer should have epoch parameter, but it produces a bug when set. So,
        num_batches = len(files_list) / config['model']['batch_size']
        for j in range(num_batches):
            _, loss_val = sess.run([train_step, cross_entropy], feed_dict={'vgg_images:0': img_batch.eval()})
            print loss_val

    coord.request_stop()
    coord.join(threads)
