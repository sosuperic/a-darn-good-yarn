# Train

import os
import tensorflow as tf
from scipy.misc import imread, imresize

from core.utils import read_yaml, get_optimizer
from core.basic_cnn import BasicVizsentCNN
from core.vgg.vgg16 import vgg16

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__cwd__ = os.path.realpath(os.getcwd())

config = read_yaml(os.path.join(__location__, 'config.yaml'))

# Setting up inputs
def get_files_labels_list():
    files_list = [os.path.join(__cwd__, 'data/Sentibank/aa_backup', f) for f in
             os.listdir(os.path.join(__cwd__, 'data/Sentibank/aa_backup')) if f.endswith('jpg')]
    labels_list = [3 for f in files_list] # temporary: 4th class in imagenet

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

def preprocess_image(image, model_arch):
    if model_arch == 'vgg':
        w, h = 224, 224
    if model_arch == 'basic_cnn':
        w, h = 227, 227
    image = tf.image.resize_image_with_crop_or_pad(image, h, w)
    image = tf.cast(image, tf.float32)
    return image

# Main setting up of graph
files_list, labels_list = get_files_labels_list()
files_tensor = tf.convert_to_tensor(files_list, dtype=tf.string)
labels_tensor = tf.convert_to_tensor(labels_list, dtype=tf.int32)
img, label = input_pipeline(files_tensor, labels_tensor)
model_arch = 'basic_cnn' #'vgg'
img = preprocess_image(img, model_arch)
img_batch, label_batch = tf.train.batch([img, label], batch_size=config['model']['batch_size'])


# Running
with tf.Session() as sess:
    # Set up training and graph
    num_classes = 1000

    # vgg = vgg16(batch_size=config['model']['batch_size'],
    #         w=config['model']['img_crop_size']['w'],
    #         h=config['model']['img_crop_size']['h'],
    #         sess=sess,
    #         load_weights=True,
    #         output_dim=num_classes)

    basic_cnn = BasicVizsentCNN(batch_size=config['model']['batch_size'], img_w=227, img_h=227, output_dim=num_classes)

    labels_onehot = tf.one_hot(label_batch, num_classes)     # (batch_size, 1000)
    optimizer = get_optimizer(config)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(basic_cnn.probs, labels_onehot))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(vgg.probs, labels_onehot))
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
            # print sess.run(basic_cnn.fc4, feed_dict={'img_batch:0': img_batch.eval()}).shape
            _, loss_val = sess.run([train_step, cross_entropy], feed_dict={'img_batch:0': img_batch.eval()})
            print loss_val

    coord.request_stop()
    coord.join(threads)
