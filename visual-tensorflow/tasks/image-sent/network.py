# Network class called by main, with train, test functions

import tensorflow as tf

from datasets import get_dataset
from core.basic_cnn import BasicVizsentCNN
from core.vgg.vgg16 import vgg16
from core.utils.utils import get_optimizer

class Network(object):
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(params)

    def train(self):
        """Train"""
        with tf.Session() as sess:
            # Get model
            model = None
            num_classes = self.dataset.get_num_labels()
            if self.params['arch'] == 'basic_cnn':
                model = BasicVizsentCNN(batch_size=self.params['batch_size'],
                                        img_w=self.params['img_crop_w'],
                                        img_h=self.params['img_crop_h'],
                                        output_dim=num_classes)
            elif self.params['arch'] == 'vgg':
                model = vgg16(batch_size=self.params['batch_size'],
                              w=self.params['img_crop_w'],
                              h=self.params['img_crop_h'],
                              sess=sess,
                              load_weights=True,
                              output_dim=num_classes)

            # Get data
            img_batch, label_batch = self.dataset.setup_graph()
            labels_onehot = tf.one_hot(label_batch, num_classes)     # (batch_size, num_classes)

            # Minimize
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.probs, labels_onehot))

            # Optimize
            optimizer = get_optimizer(self.params['optim'], self.params['lr'])
            train_step = optimizer.minimize(cross_entropy)

            # Initialize after optimization - this needs to be done after adam
            sess.run(tf.initialize_all_variables())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Training
            for i in range(self.params['epochs']):
                # Normally slice_input_producer should have epoch parameter, but it produces a bug when set. So,
                num_batches = self.dataset.get_num_batches()
                for j in range(num_batches):
                    # print sess.run(basic_cnn.fc4, feed_dict={'img_batch:0': img_batch.eval()}).shape
                    _, loss_val = sess.run([train_step, cross_entropy], feed_dict={'img_batch:0': img_batch.eval()})
                    print loss_val

            coord.request_stop()
            coord.join(threads)

