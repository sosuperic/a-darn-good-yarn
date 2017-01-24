# Network class called by main, with train, test functions

import json
import numpy as np
import os
import pickle
import tensorflow as tf

from datasets import get_dataset
from core.audio.AudioCNN import AudioCNN
from core.utils.utils import get_optimizer, load_model, save_model, setup_logging

class Network(object):
    def __init__(self, params):
        self.params = params

    ####################################################################################################################
    # Train
    ####################################################################################################################
    def train(self):
        """Train"""
        self.dataset = get_dataset(self.params)
        self.logger = self._get_logger()
        self.output_dim = self.dataset.get_output_dim()
        with tf.Session() as sess:
            # Get data
            self.logger.info('Retrieving training data and setting up graph')
            splits = self.dataset.setup_graph()
            tr_clip_batch, self.tr_label_batch = splits['train']['clip_batch'], splits['train']['label_batch']
            va_clip_batch, va_label_batch = splits['valid']['clip_batch'], splits['valid']['label_batch']

            # Get model
            model = self._get_model(sess, tr_clip_batch)

            # Loss
            self._get_loss(model)

            # Optimize - split into two steps (get gradients and then apply so we can create summary vars)
            optimizer = get_optimizer(self.params)
            grads = tf.gradients(self.loss, tf.trainable_variables())
            self.grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_step = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars)
            # capped_grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in self.grads_and_vars]
            # train_step = optimizer.apply_gradients(grads_and_vars=capped_grads_and_vars)

            # Summary ops and writer
            summary_op = self._get_summary_ops()
            tr_summary_writer = tf.summary.FileWriter(self.params['ckpt_dirpath'] + '/train', graph=tf.get_default_graph())
            va_summary_writer = tf.summary.FileWriter(self.params['ckpt_dirpath'] + '/valid')

            # Initialize after optimization - this needs to be done after adam
            coord, threads = self._initialize(sess)

            # Training
            saver = tf.train.Saver(max_to_keep=None)
            for i in range(self.params['epochs']):
                self.logger.info('Epoch {}'.format(i))
                # Normally slice_input_producer should have epoch parameter, but it produces a bug when set. So,
                num_tr_batches = self.dataset.get_num_batches('train')
                for j in range(num_tr_batches):
                    _, imgs, last_fc, loss_val, acc_val, summary = sess.run(
                        [train_step, tr_clip_batch, model.last_fc, self.loss, self.acc, summary_op])

                    self.logger.info('Train minibatch {} / {} -- Loss: {}'.format(j, num_tr_batches, loss_val))
                    self.logger.info('................... -- Acc: {}'.format(acc_val))

                    # Write summary
                    if j % 10 == 0:
                        tr_summary_writer.add_summary(summary, i * num_tr_batches + j)

                    # if j == 10:
                    #     break

                    # Save (potentially) before end of epoch just so I don't have to wait
                    if j % 100 == 0:
                        save_model(sess, saver, self.params, i, self.logger)

                # Evaluate on validation set (potentially)
                if (i+1) % self.params['val_every_epoch'] == 0:
                    num_va_batches = self.dataset.get_num_batches('valid')
                    for j in range(num_va_batches):
                        img_batch, label_batch = sess.run([va_clip_batch, va_label_batch])
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

                # Save model at end of epoch (potentially)
                save_model(sess, saver, self.params, i, self.logger)

            coord.request_stop()
            coord.join(threads)

    ####################################################################################################################
    # Test
    ####################################################################################################################
    def test(self):
        """Test"""
        self.dataset = get_dataset(self.params)
        self.logger = self._get_logger()
        self.output_dim = self.dataset.get_output_dim()
        with tf.Session() as sess:
            # Get data
            self.logger.info('Getting test set')
            te_clip_batch, self.te_label_batch = self.dataset.setup_graph()
            num_batches = self.dataset.get_num_batches('test')

            # Get model
            self.logger.info('Building graph')
            model = self._get_model(sess, te_clip_batch)

            # Loss
            self._get_loss(model)

            # Weights and gradients
            grads = tf.gradients(self.loss, tf.trainable_variables())
            self.grads_and_vars = list(zip(grads, tf.trainable_variables()))

            # Summary ops and writer
            summary_op = self._get_summary_ops()
            summary_writer = tf.summary.FileWriter(self.params['ckpt_dirpath'] + '/test', graph=tf.get_default_graph())

            # Initialize
            coord, threads = self._initialize(sess)

            # Restore model now that graph is complete -- loads weights to variables in existing graph
            self.logger.info('Restoring checkpoint')
            saver = load_model(sess, self.params)

            # print sess.run(tf.trainable_variables()[0])

            # Test
            # overall_correct = 0
            # overall_num = 0
            for j in range(num_batches):
                loss_val, acc_val, summary = sess.run([self.loss, self.acc, summary_op])

                # overall_correct += int(acc_val * te_clip_batch.get_shape().as_list()[0])
                # overall_num += te_clip_batch.get_shape().as_list()[0]
                # overall_acc = float(overall_correct) / overall_num

                self.logger.info('Test minibatch {} / {} -- Loss: {}'.format(j, num_batches, loss_val))
                # self.logger.info('................... -- Acc: {}'.format(acc_val))
                # self.logger.info('Overall acc: {}'.format(overall_acc))

                # Write summary
                if j % 10 == 0:
                    summary_writer.add_summary(summary, j)

            coord.request_stop()
            coord.join(threads)


    ####################################################################################################################
    # Predict
    ####################################################################################################################

    # def get_all_vidpaths_with_frames(self, starting_dir):
    #     """
    #     Return list of full paths to every video directory that contains frames/
    #     e.g. [<VIDEOS_PATH>/@Animated/@OldDisney/Feast/, ...]
    #     """
    #     vidpaths = []
    #     for root, dirs, files in os.walk(starting_dir):
    #         if 'frames' in os.listdir(root):
    #             vidpaths.append(root)
    #
    #     return vidpaths

    # def predict(self):
    #     """Predict"""
    #     self.logger = self._get_logger()
    #
    #     # If given path contains frames/, just predict for that one video
    #     # Else walk through directory and predict for every folder that contains frames/
    #     dirpaths = None
    #     if os.path.exists(os.path.join(self.params['vid_dirpath'], 'frames')):
    #         dirpaths = [self.params['vid_dirpath']]
    #     else:
    #         dirpaths = self.get_all_vidpaths_with_frames(self.params['vid_dirpath'])
    #
    #     for dirpath in dirpaths:
    #         # Skip if exists
    #         # if os.path.exists(os.path.join(dirpath, 'preds', 'sent_biclass_19.csv')):
    #         #     print 'Skip: {}'.format(dirpath)
    #         #     continue
    #         with tf.Session() as sess:
    #             # Get data
    #             self.logger.info('Getting images to predict for {}'.format(dirpath))
    #             self.dataset = get_dataset(self.params, dirpath)
    #             self.output_dim = self.dataset.get_output_dim()
    #             img_batch = self.dataset.setup_graph()
    #
    #             # Get model
    #             self.logger.info('Building graph')
    #             model = self._get_model(sess, img_batch)
    #
    #             # Initialize
    #             coord, threads = self._initialize(sess)
    #
    #             # Restore model now that graph is complete -- loads weights to variables in existing graph
    #             self.logger.info('Restoring checkpoint')
    #             saver = load_model(sess, self.params)
    #
    #             # Make directory to store predictions
    #             preds_dir = os.path.join(dirpath, 'preds')
    #             if not os.path.exists(preds_dir):
    #                 os.mkdir(preds_dir)
    #
    #             # Predict, write to file
    #             idx2label = self.get_idx2label()
    #             num_batches = self.dataset.get_num_batches('predict')
    #             if self.params['load_epoch'] is not None:
    #                 fn = '{}_{}.csv'.format(self.params['obj'], self.params['load_epoch'])
    #             else:
    #                 fn = '{}.csv'.format(self.params['obj'])
    #
    #             with open(os.path.join(preds_dir, fn), 'w') as f:
    #                 labels = [idx2label[i] for i in range(self.output_dim)]
    #                 f.write('{}\n'.format(','.join(labels)))
    #                 for j in range(num_batches):
    #                     last_fc, probs = sess.run([model.last_fc, model.probs],
    #                                               feed_dict={'img_batch:0': img_batch.eval()})
    #
    #                     if self.params['debug']:
    #                         print last_fc
    #                         print probs
    #                     for frame_prob in probs:
    #                         frame_prob = ','.join([str(v) for v in frame_prob])
    #                         f.write('{}\n'.format(frame_prob))
    #
    #             coord.request_stop()
    #             coord.join(threads)
    #
    #         # Clear previous video's graph
    #         tf.reset_default_graph()

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

    def _get_model(self, sess, clip_batch):
        """Return model (sess is required to load weights for vgg)"""
        # Get model
        model = None
        self.logger.info('Making {} model'.format(self.params['arch']))
        if self.params['arch'] == 'cnn':
            model = AudioCNN(
                # batch_size=self.params['batch_size'],
                output_dim=self.output_dim,
                clips=clip_batch,
                dropout_keep=self.params['dropout'])

        return model

    def _get_loss(self, model):
        label_batch = self.tr_label_batch if self.params['mode'] == 'train' else self.te_label_batch
        label_batch_op = tf.placeholder_with_default(label_batch, shape=[self.params['batch_size']],
                                                     name='label_batch')
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.last_fc, label_batch_op))))

    def _get_summary_ops(self):
        """Define summaries and return summary_op"""
        self.loss_summary = tf.summary.scalar('loss', self.loss)

        # Weights and gradients. TODO: why doesn't this work for test? Type error
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in self.grads_and_vars:
            tf.summary.histogram(var.op.name+'/gradient', grad)

        summary_op = tf.summary.merge_all()
        return summary_op

    def _initialize(self, sess):
        if self.params['mode'] == 'train':
            sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        return coord, threads
