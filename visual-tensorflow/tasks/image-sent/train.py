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
print config

# Get Sentibank data
files_list = [os.path.join(__cwd__, 'data/Sentibank/aa_backup', f) for f in
         os.listdir(os.path.join(__cwd__, 'data/Sentibank/aa_backup')) if f.endswith('jpg')]
labels_list = [1 for f in files_list]
print files_list
# files_space_labels = [f + " label" for f in files]


from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
files_tensor = ops.convert_to_tensor(files_list, dtype=dtypes.string)
labels_tensor = ops.convert_to_tensor(labels_list, dtype=dtypes.int32)
# files_tensor = tf.convert_to_tensor(files_list, dtype=tf.string)
# labels_tensor = tf.convert_to_tensor(labels_list, dtype=tf.int32)

def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image.set_shape([256, 256, 3])
    # import numpy as np
    # image = tf.convert_to_tensor(np.random.random([256,256,3]))
    return image, label

# Put all this in a function input_pipeline():
input_queue = tf.train.slice_input_producer([files_tensor, labels_tensor],
                                            # num_epochs=10, WTFFFF IS THIS BUGGG
                                            shuffle=False,
                                            capacity=32
                                            )
image, label = read_images_from_disk(input_queue)

# Preprocessing
# image = preprocess_image
image = tf.image.resize_images(image, 224, 224)
# label = preprocess_label

# print image
# print label

batch_size = 4
# min_after_dequeue = 10000
# capacity = min_after_dequeue + 3 * batch_size
# image_batch, label_batch = tf.train.batch(
#     [image, label],
#     batch_size=batch_size,
#     capacity=capacity,
#     min_after_dequeue=min_after_dequeue)

# image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print image.eval()

    # img = image.eval()
    # sess.run([image, label])
    # sess.run([label_batch])

    coord.request_stop()
    coord.join(threads)


#########################################################################################################
#
# # Set up pipeline
# def read_my_file_format(filename_queue):
#
# # This method: http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
# # def read_my_file_format(filename_and_label_tensor):
#     reader = tf.WholeFileReader()
#
#     # Fixed: https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/models/image/cifar10/cifar10_input.py
#     # record_bytes = (256 * 256 * 3) + 1
#     # reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
#
#     filename, image = reader.read(filename_queue)
#
#     # filename, label = tf.decode_csv(filename_and_label_tensor, [[""], [""]], " ")
#     # image = tf.read_file(filename)
#     image = tf.image.decode_jpeg(image, channels=3)
#
#     # label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)     # start index, how many bytes
#
#     # Convert filename to label?
#     label = 1
#
#     # print image
#     # image = tf.reshape(image, [256, 256, 3])
#     # image.set_shape([256, 256, 3])
#
#     # image.set_shape([256, 256, 3])
#
#     return image, label
#
#     # label_bytes =
#     # reader = tf.Fixed
#     #
#     # # Convert filenames to polarities, emotions, whatever for labels
#     # # print filename
#     # label = 1
#     #
#     # # Convert image to
#     # image = tf.image.decode_jpeg(image, channels=3)
#     # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#     # # print image
#     # # image.set_shape([256, 256, 3])
#     # # image = tf.cast(image, tf.float32)
#     # # image = tf.image.resize_images(image, 224, 224)
#     # # image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
#     # # print image
#     #
#     # # Question: what happens if there's an error? Will batching simply get next record in queue?
#     # return image, label
#
# def input_pipeline(filenames, batch_size=32, num_epochs=10):
#     filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs) #  list of files to read
#     image, label = read_my_file_format(filename_queue)
#     # image, label = read_my_file_format(filename_queue.dequeue())
#
#     min_after_dequeue = 10
#     capacity = min_after_dequeue + 3 * batch_size
#     image_batch, label_batch = tf.train.shuffle_batch(
#         [image, label],
#         batch_size=batch_size,
#         capacity=capacity,
#         min_after_dequeue=min_after_dequeue)
#     return image_batch, label_batch
#
#
#
# # Start session
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     batch_size = 32
#     filenames, images = input_pipeline(files, batch_size=batch_size)
#     # filenames, images = input_pipeline(files_space_labels, batch_size=batch_size)
#
#     # vgg = vgg16(images, sess=sess)
#     # y_pred = vgg.probs
#     # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)
#     # train_op = tf.train.AdamOptimizer().minimize(loss)
#
#     print images
#     print images.get_shape()
#     sess.run([images])
#     # sess.run([vgg.probs])
#
#
#     # vgg = vgg16(imgs, sess=sess)
#
#     coord.request_stop()
#     coord.join(threads)

# print config
# Load fine-tuned VGG
# sess = tf.Session()
# imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
# vgg = vgg16(imgs, sess=sess)
#
# sample_im_path = os.path.realpath(os.path.join(os.getcwd(), 'core/vgg/laska.png'))
# img1 = imread(sample_im_path, mode='RGB')
# img1 = imresize(img1, (224, 224))
#
# prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
# preds = (np.argsort(prob)[::-1])[0:5]
# for p in preds:
#     print class_names[p], prob[p]