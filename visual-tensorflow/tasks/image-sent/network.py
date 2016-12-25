# Network class called by main, with train, test functions

import json
import numpy as np
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

    ####################################################################################################################
    # Train
    ####################################################################################################################
    def train(self):
        """Train"""
        self.logger = self._get_logger()
        self.output_dim = self.dataset.get_output_dim()
        with tf.Session() as sess:
            # Get data
            self.logger.info('Retrieving training data and setting up graph')
            splits = self.dataset.setup_graph()
            tr_img_batch, self.tr_label_batch = splits['train']['img_batch'], splits['train']['label_batch']
            va_img_batch, va_label_batch = splits['valid']['img_batch'], splits['valid']['label_batch']

            # Get model
            model = self._get_model(sess, tr_img_batch)

            # Loss
            self._get_loss(model)

            # Optimize - split into two steps (get gradients and then apply so we can create summary vars)
            optimizer = get_optimizer(self.params['optim'], self.params['lr'])
            grads = tf.gradients(self.loss, tf.trainable_variables())
            self.grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_step = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars)

            # Summary ops and writer
            summary_op = self._get_summary_ops()
            tr_summary_writer = tf.summary.FileWriter(self.params['save_dir'] + '/train', graph=tf.get_default_graph())
            va_summary_writer = tf.summary.FileWriter(self.params['save_dir'] + '/valid')

            # Initialize after optimization - this needs to be done after adam
            coord, threads = self._initialize(sess)

            # Training
            saver = tf.train.Saver(max_to_keep=None)
            for i in range(self.params['epochs']):
                self.logger.info('Epoch {}'.format(i))
                # Normally slice_input_producer should have epoch parameter, but it produces a bug when set. So,
                num_tr_batches = self.dataset.get_num_batches('train')
                for j in range(num_tr_batches):
                    _, loss_val, acc_val, summary = sess.run([train_step, self.loss, self.acc, summary_op])
                                                             # feed_dict={'class_weights:0': label2count})

                    self.logger.info('Train minibatch {} / {} -- Loss: {}'.format(j, num_tr_batches, loss_val))
                    self.logger.info('................... -- Acc: {}'.format(acc_val))

                    # # Write summary
                    if j % 10 == 0:
                        tr_summary_writer.add_summary(summary, i * num_tr_batches + j)

                    # if j == 10:
                    #     break

                # Evaluate on validation set (potentially)
                if (i+1) % self.params['val_every_epoch'] == 0:
                    num_va_batches = self.dataset.get_num_batches('valid')
                    for j in range(num_va_batches):
                        img_batch, label_batch = sess.run([va_img_batch, va_label_batch])
                        loss_val, acc_val, loss_summary, acc_summary = sess.run([self.loss, self.acc,
                                                                    self.loss_summary, self.acc_summary],
                                                              feed_dict={'img_batch:0': img_batch,
                                                                        'label_batch:0': label_batch})

                        self.logger.info('Valid minibatch {} / {} -- Loss: {}'.format(j, num_va_batches, loss_val))
                        self.logger.info('................... -- Acc: {}'.format(acc_val))

                        # Write summary
                        if j % 10 == 0:
                            va_summary_writer.add_summary(loss_summary, i * num_tr_batches + j)
                            va_summary_writer.add_summary(acc_summary, i * num_tr_batches + j)

                        # if j == 5:
                        #     break

                # Save model at end of epoch (potentially)
                save_model(sess, saver, self.params, i, self.logger)

            coord.request_stop()
            coord.join(threads)

    ####################################################################################################################
    # Test
    ####################################################################################################################
    def test(self):
        """Test"""
        self.logger = self._get_logger()
        self.output_dim = self.dataset.get_output_dim()
        with tf.Session() as sess:
            # Get model
            self.logger.info('Load model')
            saver = load_model(sess, self.params)
            # print sess.run(tf.trainable_variables()[0])

            # Get data
            self.logger.info('Getting test set')
            splits = self.dataset.setup_graph()
            te_img_batch, te_label_batch = splits['test']['img_batch'], splits['test']['label_batch']

            # Loss
            self._get_loss(saver)

            # Summary ops and writer
            summary_op = self._get_summary_ops()
            summary_writer = tf.summary.FileWriter(self.params['save_dir'] + '/test', graph=tf.get_default_graph())

            # Initialize
            coord, threads = self._initialize(sess)

            # Test
            num_batches = self.dataset.get_num_batches('test')
            for j in range(num_batches):
                loss_val, acc_val, summary = sess.run([self.loss, self.acc, summary_op],
                                                      feed_dict={'img_batch:0': te_img_batch.eval(),
                                                                 'label_batch:0': te_label_batch.eval()})

                self.logger.info('Test minibatch {} / {} -- Loss: {}'.format(j, num_batches, loss_val))
                self.logger.info('................... -- Acc: {}'.format(acc_val))

                # Write summary
                if j % 1 == 0:
                    summary_writer.add_summary(summary, j)


            coord.request_stop()
            coord.join(threads)

    ####################################################################################################################
    # Helper functions
    ####################################################################################################################
    def _get_logger(self):
        """Return logger, where path is dependent on mode (train/test), arch, and obj"""
        logs_path = os.path.join(os.path.dirname(__file__), 'logs')
        _, logger = setup_logging(save_path=os.path.join(
            logs_path, '{}-{}-{}.log'.format(self.params['mode'], self.params['arch'], self.params['obj'])))
        # _, self.logger = setup_logging(save_path=logs_path)
        return logger

    def _get_model(self, sess, img_batch):
        """Return model (sess is required to load weights for vgg)"""
        # Get model
        model = None
        self.logger.info('Making {} model'.format(self.params['arch']))
        if self.params['arch'] == 'basic_cnn':
            model = BasicVizsentCNN(batch_size=self.params['batch_size'],
                                    img_w=self.params['img_crop_w'],
                                    img_h=self.params['img_crop_h'],
                                    output_dim=self.output_dim,
                                    imgs=img_batch)
        elif 'vgg' in self.params['arch']:
            load_weights = True if self.params['arch'] == 'vgg_finetune' else False
            model = vgg16(batch_size=self.params['batch_size'],
                          w=self.params['img_crop_w'],
                          h=self.params['img_crop_h'],
                          sess=sess,
                          load_weights=load_weights,
                          output_dim=self.output_dim,
                          img_batch=img_batch)
        return model

    def _get_loss(self, model):
        # Get class weights
        # Formula: Divide each count into max count, the normalize:
        # Example: [35, 1, 5] -> [1, 3.5, 7] -> [0.08696, 0.30435, 0.608696]
        label2count = json.load(open(os.path.join(self.params['save_dir'], 'label2count.json'), 'r'))
        label2count = [float(c) for l,c in label2count.items()]             # (num_classes, )
        self.logger.info('Class counts: {}'.format(label2count))
        for i, c in enumerate(label2count):
            if c != 0:
                label2count[i] = max(label2count) / c
            else:
                label2count[i] = 0.0
        label2count = [w / sum(label2count) for w in label2count]
        self.logger.info('Class weights: {}'.format(label2count))
        label2count = np.array(label2count)
        label2count = np.expand_dims(label2count, 1).transpose()            # (num_classes, 1) -> (1, num_classes)
        class_weights = tf.cast(tf.constant(label2count), tf.float32)

        label_batch_op = tf.placeholder_with_default(self.tr_label_batch, shape=[self.params['batch_size']],
                                                     name='label_batch')

        if self.params['obj'] == 'sent_reg':
            # TODO: add weights for regression as well
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.last_fc, label_batch_op))))
        else:
            labels_onehot = tf.one_hot(label_batch_op, self.output_dim)     # (batch_size, num_classes)
            weighted_logits = tf.mul(model.probs, class_weights)            # (batch_size, num_classes)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(weighted_logits, labels_onehot))

            # Accuracy
            acc = tf.equal(tf.cast(tf.argmax(model.probs, 1), tf.int32), label_batch_op)
            self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    def _get_summary_ops(self):
        """Define summaries and return summary_op"""
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        if self.params['obj'] != 'sent_reg':    # classification, thus has accuracy
            self.acc_summary = tf.summary.scalar('accuracy', self.acc)

        # Weights and gradients
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in self.grads_and_vars:
            tf.summary.histogram(var.op.name+'/gradient', grad)
        summary_op = tf.summary.merge_all()
        return summary_op

    def _initialize(self, sess):
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        return coord, threads