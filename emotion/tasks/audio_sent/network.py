# Network class called by main, with train, test functions

from collections import defaultdict
import cPickle as pickle
import librosa
import math
import numpy as np
import os
import tensorflow as tf
import time

from core.audio.AudioCNN import AudioCNN
from core.utils.utils import get_optimizer, load_model, save_model, setup_logging
from datasets import get_dataset, MELGRAM_20S_SIZE, NUMPTS_AND_MEANSTD_PATH
from prepare_data import compute_log_melgram_from_np, N_FFT, N_MELS, HOP_LEN

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
        # self.output_dim = self.dataset.get_output_dim()
        with tf.Session() as sess:
            # Get data
            self.logger.info('Retrieving training data and setting up graph')
            splits = self.dataset.setup_graph()
            tr_clip_batch, self.tr_label_batch = splits['train']['clip_batch'], splits['train']['label_batch']
            va_clip_batch, va_label_batch = splits['valid']['clip_batch'], splits['valid']['label_batch']

            # # Get model
            model = self._get_model(sess, tr_clip_batch, self.params['bn_decay'], is_training=True)

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

            # clips = sess.run(tr_clip_batch)
            # print clips[0], clips[0].shape

            # Training
            saver = tf.train.Saver(max_to_keep=None)
            for i in range(self.params['epochs']):
                self.logger.info('Epoch {}'.format(i))
                # Normally slice_input_producer should have epoch parameter, but it produces a bug when set. So,
                num_tr_batches = self.dataset.get_num_batches('train')
                for j in range(num_tr_batches):
                    _, imgs, out, loss_val, summary = sess.run(
                        [train_step, tr_clip_batch, model.out, self.loss, summary_op])

                    self.logger.info('Train minibatch {} / {} -- Loss: {}'.format(j, num_tr_batches, loss_val))

                    # Write summary
                    if j % 10 == 0:
                        tr_summary_writer.add_summary(summary, i * num_tr_batches + j)

                    # if j == 5:
                    #     break

                    # Save (potentially) before end of epoch just so I don't have to wait
                    if j % 100 == 0:
                        save_model(sess, saver, self.params, i, self.logger)

                # Evaluate on validation set (potentially)
                if (i+1) % self.params['val_every_epoch'] == 0:
                    num_va_batches = self.dataset.get_num_batches('valid')
                    for j in range(num_va_batches):
                        clip_batch, label_batch = sess.run([va_clip_batch, va_label_batch])
                        loss_val, loss_summary = sess.run(
                            [self.loss, self.loss_summary],
                            feed_dict={'clip_batch:0': clip_batch, 'label_batch:0': label_batch})

                        self.logger.info('Valid minibatch {} / {} -- Loss: {}'.format(j, num_va_batches, loss_val))

                        # Write summary
                        if j % 10 == 0:
                            va_summary_writer.add_summary(loss_summary, i * num_tr_batches + j)

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
        # self.output_dim = self.dataset.get_output_dim()
        with tf.Session() as sess:
            # Get data
            self.logger.info('Getting test set')
            te_clip_batch, self.te_label_batch = self.dataset.setup_graph()
            num_batches = self.dataset.get_num_batches('test')

            # Get model
            self.logger.info('Building graph')
            model = self._get_model(sess, te_clip_batch, self.params['bn_decay'], is_training=False)
            # model = self._get_model(sess, te_clip_batch, self.params['bn_decay'], is_training=True)

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
            for j in range(num_batches):
                fc, out, loss_val, summary = sess.run([model.fc, model.out, self.loss, summary_op])
                self.logger.info('Test minibatch {} / {} -- Loss: {}'.format(j, num_batches, loss_val))

                # print fc, out
                # Write summary
                # if j % 10 == 0:
                #     summary_writer.add_summary(summary, j)

            coord.request_stop()
            coord.join(threads)


    ####################################################################################################################
    # Predict
    ####################################################################################################################

    def get_all_vidpaths_with_mp3(self, starting_dir):
        """
        Return list of paths to every mp3 file (which contains the audio for the video)
        e.g. [data/videos/films/animated/The Incredibles (2004)/The Incredibles (2004).mp3, ...]
        """
        self.logger.info('Getting all vidpaths with mp3')
        mp3paths = []
        for root, dirs, files in os.walk(starting_dir):
            for f in files:
                if f.endswith('mp3'):
                    mp3paths.append(os.path.join(root, f))
        return mp3paths

    def predict(self):
        """Predict"""
        self.logger = self._get_logger()

        # If given path contains an mp3 file, just predict for that one video
        # Else walk through directory and predict for every folder that contains an mp3 file
        mp3paths = None
        mp3_in_vid_dirpath = ['mp3' in f for f in os.listdir(self.params['vid_dirpath'])]
        if True in mp3_in_vid_dirpath:
            mp3_f = os.listdir(self.params['vid_dirpath'])[mp3_in_vid_dirpath.index(True)]
            mp3paths = [os.path.join(self.params['vid_dirpath'], mp3_f)]
        else:
            mp3paths = self.get_all_vidpaths_with_mp3(self.params['vid_dirpath'])

        # Load mean and std
        mean, std = self.load_meanstd()

        for mp3path in mp3paths:
            # Skip if exists
            # if os.path.exists(os.path.join(dirpath, 'preds', 'sent_biclass_19.csv')):
            #     print 'Skip: {}'.format(dirpath)
            #     continue
            start_time = time.time()
            with tf.Session() as sess:
                # Get data
                self.logger.info('Loading mp3 to predict for {}'.format(mp3path))
                src, sr = librosa.load(mp3path, sr=None)    # uses default sample rate, which should be 12000

                # Calculate number of points given 1-D source signal
                # Example: (l-w)/s + 1; l = 101, w = 20, s = 10
                src_nsec = len(src) / sr
                melgram_nsec = 20
                num_pts = ((src_nsec - melgram_nsec) / float(self.params['stride'])) + 1      # 20 for 20 second melgram
                num_pts = int(math.floor(num_pts))
                num_batches = int(math.floor(float(num_pts) / self.params['batch_size']))

                # Get model
                self.logger.info('Creating dummy input and getting model')
                # Feed in constant tensor as clip_batch that will get overridden by feed_dict
                batch_shape = [self.params['batch_size']] + MELGRAM_20S_SIZE + [1]
                with tf.variable_scope('dummy_input'):
                    self.clip_batch = tf.zeros(batch_shape)
                # model = self._get_model(sess, self.clip_batch, self.params['bn_decay'], is_training=True)
                model = self._get_model(sess, self.clip_batch, self.params['bn_decay'], is_training=False)

                # Initialize
                coord, threads = self._initialize(sess)

                # Restore model now that graph is complete -- loads weights to variables in existing graph
                self.logger.info('Restoring checkpoint')
                saver = load_model(sess, self.params)

                # Make directory to store predictions
                preds_dir = os.path.join(os.path.dirname(mp3path), 'preds')
                if not os.path.exists(preds_dir):
                    os.mkdir(preds_dir)

                # Get file to write predictions
                if self.params['load_epoch'] is not None:
                    fn = 'audio-{}_{}.csv'.format(self.params['obj'], self.params['load_epoch'])
                else:
                    fn = 'audio-{}.csv'.format(self.params['obj'])

                # Predict
                s2preds = defaultdict(list)
                for j in range(num_batches):
                    cur_batch = np.zeros(batch_shape)
                    for k in range(self.params['batch_size']):
                        src_start_idx = (j * k) * (sr * self.params['stride'])
                        src_end_idx = (j * (k+1)) * (sr * self.params['stride'])
                        cur_src = src[src_start_idx:src_end_idx]
                        cur_melgram = compute_log_melgram_from_np(cur_src, melgram_nsec, sr, HOP_LEN, N_FFT, N_MELS)
                        cur_melgram = (cur_melgram - mean) / std
                        cur_batch[k] = np.expand_dims(cur_melgram, 2)

                    cur_batch = cur_batch.astype(np.float32, copy=False)
                    fc, outs = sess.run([model.fc, model.out], feed_dict={'clip_batch:0': cur_batch})

                    if self.params['debug']:
                        # print fc, outs
                        print outs

                    batch_start_s = (j * self.params['batch_size'] * self.params['stride'])      # second
                    for k, out in enumerate(outs):
                        cur_melgram_s = batch_start_s + (k * self.params['stride'])
                        for rel_s in range(melgram_nsec):
                            cur_s = cur_melgram_s + rel_s
                            # print batch_start_s, cur_melgram_s, rel_s, cur_s
                            s2preds[cur_s].append(out[0])

                with open(os.path.join(preds_dir, fn), 'wb') as f:
                    f.write('Valence\n')
                    for s in sorted(s2preds):
                        # print s, s2preds[s]
                        pred = sum(s2preds[s]) / float(len(s2preds[s]))
                        f.write('{}\n'.format(pred))

                print 'Number pts according to source signal: {}'.format(num_pts)
                print 'Number batches according to source signal: {}'.format(num_batches)
                print 'Length of source signal in seconds: {}'.format(src_nsec)
                print 'Time elapsed: {} minutes'.format((time.time() - start_time) / 60.0)

                coord.request_stop()
                coord.join(threads)

            # Clear previous video's graph
            tf.reset_default_graph()

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

    def _get_model(self, sess, clip_batch, bn_decay, is_training=None):
        """Return model (sess is required to load weights for vgg)"""
        # Get model
        model = None
        self.logger.info('Making {} model'.format(self.params['arch']))
        if self.params['arch'] == 'cnn':
            model = AudioCNN(
                # output_dim=self.output_dim,
                clips=clip_batch,
                bn_decay=bn_decay,
                is_training=is_training)

        return model

    def _get_loss(self, model):
        label_batch = self.tr_label_batch if self.params['mode'] == 'train' else self.te_label_batch
        label_batch_op = tf.placeholder_with_default(label_batch, shape=[self.params['batch_size']],
                                                     name='label_batch')
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.out, label_batch_op))))

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

    def load_meanstd(self):
        """
        Load the pre-computed number of points per train-valid-test split and per mel-bin mean and stddev so we can
        normalize data.
        """
        numpts_and_meanstd = pickle.load(open(NUMPTS_AND_MEANSTD_PATH, 'rb'))
        mean = numpts_and_meanstd['mean']      # (number of mel-bins, 1)
        std = numpts_and_meanstd['std']        # (number of mel-bins, 1)
        return mean, std
