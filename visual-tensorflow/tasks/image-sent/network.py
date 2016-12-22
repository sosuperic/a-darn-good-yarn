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
            self.logger.info('Retrieving training data and setting up graph')
            splits = self.dataset.setup_graph()
            tr_img_batch, tr_label_batch = splits['train']['img_batch'], splits['train']['label_batch']
            va_img_batch, va_label_batch = splits['train']['img_batch'], splits['train']['label_batch']

            # Loss
            if self.params['obj'] == 'sent_reg':
                loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.last_fc, tr_label_batch))))
            else:
                labels_onehot = tf.one_hot(tr_label_batch, output_dim)     # (batch_size, num_classes)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.probs, labels_onehot))

                # Accuracy
                acc = tf.equal(tf.cast(tf.argmax(model.probs, 1), tf.int32), tr_label_batch)
                acc = tf.reduce_mean(tf.cast(acc, tf.float32))

            # Optimize - split into two steps (get gradients and then apply so we can create summary vars)
            optimizer = get_optimizer(self.params['optim'], self.params['lr'])
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_step = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

            # Summary ops
            tf.summary.scalar('loss', loss)
            if self.params['obj'] != 'sent_reg':    # classification, thus has accuracy
                tf.summary.scalar('accuracy', acc)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            for grad, var in grads_and_vars:
                print var.op.name, var.op.name is None, grad is None
                tf.summary.histogram(var.op.name+'/gradient', grad)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.params['save_dir'], graph=tf.get_default_graph())

            # Initialize after optimization - this needs to be done after adam
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Training
            saver = tf.train.Saver(max_to_keep=None)
            for i in range(self.params['epochs']):
                self.logger.info('Epoch {}'.format(i))
                # Normally slice_input_producer should have epoch parameter, but it produces a bug when set. So,
                num_batches = self.dataset.get_num_batches('train')
                for j in range(num_batches):
                    _, loss_val, acc_val, summary = sess.run([train_step, loss, acc, summary_op], feed_dict={'img_batch:0': tr_img_batch.eval()})
                    self.logger.info('Minibatch {} / {} -- Loss: {}'.format(j, num_batches, loss_val))
                    self.logger.info('................... -- Acc: {}'.format(acc_val))

                    # Write summary
                    summary_writer.add_summary(summary, i * num_batches + j)

                # Evaluate on validation set (potentially)
                num_batches = self.dataset.get_num_batches('valid')
                for j in range(num_batches):
                    loss_val = sess.run(loss, feed_dict={'img_batch:0': va_img_batch.eval()})
                    self.logger.info('Minibatch {} / {} -- Loss: {}'.format(j, num_batches, loss_val))


                # Save model at end of epoch (potentially)
                save_model(sess, saver, self.params, i, self.logger)

            coord.request_stop()
            coord.join(threads)


    def test(self):
        """Test"""
        with tf.Session() as sess:
            print 'Load model'

            saver = load_model(sess, self.params)

            # print sess.run(tf.trainable_variables()[0])