# Network class called by main, with train, test functions

import os
import tensorflow as tf

from datasets import get_dataset
from core.basic_cnn import BasicVizsentCNN
from core.vgg.vgg16 import vgg16
from core.utils.utils import get_optimizer, load_model, save_model, setup_logging

class Network(object):
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(params)

    def train(self):
        """Train"""
        logs_path = os.path.join(os.path.dirname(__file__), 'logs')
        _, self.logger = setup_logging(save_path=os.path.join(logs_path, 'train.log'))
        # _, self.logger = setup_logging(save_path=logs_path)
        with tf.Session() as sess:
            # Get model
            model = None
            output_dim = self.dataset.get_output_dim()
            self.logger.info('Making {} model'.format(self.params['arch']))
            if self.params['arch'] == 'basic_cnn':
                model = BasicVizsentCNN(batch_size=self.params['batch_size'],
                                        img_w=self.params['img_crop_w'],
                                        img_h=self.params['img_crop_h'],
                                        output_dim=output_dim)
            elif 'vgg' in self.params['arch']:
                load_weights = True if self.params['arch'] == 'vgg_finetune' else False
                model = vgg16(batch_size=self.params['batch_size'],
                              w=self.params['img_crop_w'],
                              h=self.params['img_crop_h'],
                              sess=sess,
                              load_weights=load_weights,
                              output_dim=output_dim)

            # Get data
            img_batch, label_batch = self.dataset.setup_graph()

            # Loss
            if self.params['obj'] == 'sent_reg':
                loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.last_fc, label_batch))))
            else:
                labels_onehot = tf.one_hot(label_batch, output_dim)     # (batch_size, num_classes)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.probs, labels_onehot))

            # Optimize
            optimizer = get_optimizer(self.params['optim'], self.params['lr'])
            train_step = optimizer.minimize(loss)

            # Initialize after optimization - this needs to be done after adam
            sess.run(tf.initialize_all_variables())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Training
            saver = tf.train.Saver(max_to_keep=None)
            for i in range(self.params['epochs']):
                # Normally slice_input_producer should have epoch parameter, but it produces a bug when set. So,
                num_batches = self.dataset.get_num_batches()
                for j in range(num_batches):
                    # print sess.run(basic_cnn.fc4, feed_dict={'img_batch:0': img_batch.eval()}).shape
                    _, loss_val = sess.run([train_step, loss], feed_dict={'img_batch:0': img_batch.eval()})
                    print loss_val
                    # break

                # Save model at end of epoch (potentially)
                save_model(sess, saver, self.params, i)

            coord.request_stop()
            coord.join(threads)


    def test(self):
        """Test"""
        with tf.Session() as sess:
            print 'Load model'

            saver = load_model(sess, self.params)

            # print sess.run(tf.trainable_variables()[0])