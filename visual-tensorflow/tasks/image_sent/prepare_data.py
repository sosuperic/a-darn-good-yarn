# Prepare the data (parse, convert to tfrecords, download, etc.)

import argparse
from collections import defaultdict, Counter
from fuzzywuzzy import fuzz
import io
import json
import numpy as np
import os
import pickle
from PIL import Image
from pprint import pprint
import re
import sqlite3
import subprocess
import tensorflow as tf
import urllib

from core.utils.utils import read_yaml
from core.utils.MovieReader import MovieReader
from core.utils.CreditsLocator import CreditsLocator


### SENTIBANK + MVSO
# Sentibank - bi_concepts1553: mapping ajdective noun pairs to sentiment
SENTIBANK_FLICKR_PATH = 'data/Sentibank/Flickr/'
SENTIBANK_BC_PATH = 'data/Sentibank/Flickr/bi_concepts1553'
EMOLEX_PATH = 'data/emolex/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'

# MVSO dataset - mutlilingual, larger version of Sentibank VSO; also has emotions
MVSO_PATH = 'data/MVSO'
MVSO_BC_PATH = 'data/MVSO/imgs/'

# Labels
SENT_BICLASS_LABEL2INT = {'neg': 0, 'pos': 1}
SENT_TRICLASS_LABEL2INT = {'neg':0, 'neutral':1, 'pos':2}
SENTIBANK_EMO_LABEL2INT = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3,
             'joy': 4, 'sadness': 5, 'surprise': 6, 'trust': 7}
MVSO_EMO_LABEL2INT = {'ecstasy': 0, 'joy': 1, 'serenity': 2,
                      'admiration': 3, 'trust': 4, 'acceptance': 5,
                      'terror': 6, 'fear': 7, 'apprehension': 8,
                      'amazement': 9, 'surprise': 10, 'distraction': 11,
                      'grief': 12, 'sadness': 13, 'pensiveness': 14,
                      'loathing': 15, 'disgust': 16, 'boredom': 17,
                      'rage': 18, 'anger': 19, 'annoyance': 20,
                      'vigilance': 21, 'anticipation': 22, 'interest': 23}

### OTHER
# You dataset - 20k images with emotions
YOU_IMEMO_PATH = 'data/you_imemo/agg'

# Plutchik
PLUTCHIK_PATH = 'data/plutchik'

# Videos path
VIDEOS_PATH = 'data/videos'

# CMU Movie Summary path
CMU_PATH = 'data/CMU_movie_summary/MovieSummaries/'

# Videos
VID_EXTS = ['webm', 'mkv', 'flv', 'vob', 'ogv', 'ogg', 'drc', 'gif', 'gifv', 'mng', 'avi', 'mov', 'qt', 'wmv',
                'yuv', 'rm', 'rmvb', 'asf', 'amv', 'mp4', 'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'm2v',
                'm4v', 'svi', '3gp', '3g2', 'mxf', 'roq', 'nsv', 'flv', 'f4v', 'f4p', 'f4a', 'f4b']
VIDEOPATH_DB = 'data/db/VideoPath.db'
VIDEOMETADATA_DB = 'data/db/VideoMetadata.pkl'

########################################################################################################################
# Sentibank
########################################################################################################################
# Getting data structures for each objective
def get_Sentibank_bc2sent():
    """
    Return dictionary mapping bi_concept to positive-negative polarity values

    Example line: frail_hand [sentiment: -1.44] [#imgs: 358]
        - note: number of imgs is number returned by Flickr - not necessarily what's in dataset
    """
    bc2sent_and_count = {}
    with open('data/Sentibank/VSO/3244ANPs.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.endswith(']') and '_' in line:
                m = re.match(r'(.+_.+)\s\[sentiment:\s(.+)\]\s\[#imgs:\s(.+)\]', line)
                bc, sent, _ = m.group(1), float(m.group(2)), int(m.group(3).replace(',', ''))
                bc2sent_and_count[bc] = sent
    return bc2sent_and_count

def get_Sentibank_bc2emo():
    """
    Use emolex to map bi_concept to emotions. Return dict with bc as key, counts of emotions as values.

    Stats: 857 bc's with at least one emotion (57.3%) , 696 emotions without any emotions
    """
    def get_emolex():
        word2emotions = defaultdict(set)
        f = open(EMOLEX_PATH, 'rb')
        i = 0
        for line in f.readlines():
            if i > 45:          # Previous lines are readme
                word, emotion, flag = line.strip('\n').split()
                if emotion == 'positive' or emotion == 'negative':
                    continue
                if int(flag) == 1:
                    word2emotions[word].add(emotion)
            i += 1
        return word2emotions

    bc2emo = defaultdict(list)
    bc2img_fps = get_all_VSO_img_fps('Sentibank')
    word2emotions = get_emolex()
    for bc, _ in bc2img_fps.items():
        adj, noun = bc.split('_')
        if adj in word2emotions:
            for emotion in word2emotions[adj]:
                bc2emo[bc].append(emotion)
        if noun in word2emotions:
            for emotion in word2emotions[noun]:
                bc2emo[bc].append(emotion)
        bc2emo[bc] = Counter(bc2emo[bc])

    return bc2emo

def get_Sentibank_bc2idx():
    """Return dictionary mapping biconcept to idx"""
    bc2idx = {}
    for i, bc in enumerate(sorted([d for d in os.listdir(SENTIBANK_BC_PATH) if not d.startswith('.')])):
        bc2idx[bc] = i
    return bc2idx

########################################################################################################################
# MVSO
########################################################################################################################
def get_MVSO_bc2sent():
    """Return dict from bi_concept to sentiment value"""
    bc2sentiment = {}
    with open(os.path.join(MVSO_PATH, 'mvso_sentiment', 'english.csv'), 'r') as f:
        for line in f.readlines():
            bc, sentiment = line.strip().split(',')
            bc2sentiment[bc] = float(sentiment)
    return bc2sentiment

def get_MVSO_bc2emo2val():
    """Return dict from bi_concept to dict from emotion to score"""
    bc2emotion2value = defaultdict(dict)
    col2emo = {}
    with open(os.path.join(MVSO_PATH, 'ANP_emotion_scores', 'ANP_emotion_mapping_english.csv'), 'r') as f:
        i = 0
        for line in f.readlines():
            if i == 0:      # header
                header = line.strip().split(',')
                for j in range(1, len(header)):
                    col2emo[j] = header[j]
                i += 1
            else:
                line = line.strip().split(',')
                bc = line[0]
                for j in range(1, len(line)):
                    emotion = col2emo[j]
                    bc2emotion2value[bc][emotion] = float(line[j])
                i += 1

    return bc2emotion2value

def get_MVSO_bc2emo():
    """Filter result of get_MVSO_bc2emotion2value by mapping bc to max emo"""
    bc2emo2val = get_MVSO_bc2emo2val()
    bc2emo = {}
    for bc, emo2val in bc2emo2val.items():
        bc2emo[bc] = max(emo2val.keys(), key=(lambda key: emo2val[key]))
    return bc2emo

def get_MVSO_bc2idx():
    """Return dictionary mapping biconcept to idx"""
    bc2idx = {}
    for i, bc in enumerate(sorted([d for d in os.listdir(MVSO_BC_PATH) if not d.startswith('.')])):
        bc2idx[bc] = i
    return bc2idx

def download_MVSO_imgs(output_dir=os.path.join(MVSO_PATH, 'imgs'), target_w=256, target_h=256):
    """Download, resize, and center crop images"""
    import socket
    socket.setdefaulttimeout(30)

    mr = MovieReader()                  # used to resize and center crop

    def retrieve_img_and_process(url_and_fp):
        url, fp = url_and_fp[0], url_and_fp[1]
        urllib.urlretrieve(url, fp)

        # Reopen image to resize and central crop
        try:
            im = Image.open(fp)
            if im.mode != 'RGB':        # type L, P, etc. shows some type of Flickr unavailable photo img
                os.remove(fp)
                # continue
            im = np.array(im)
            im =  mr.resize_and_center_crop(im, target_w, target_h)
            Image.fromarray(im).save(fp)
        except Exception as e:
            # print url
            # print e
            pass

    # import time

    urls = []
    fps = []
    with open(os.path.join(MVSO_PATH, 'image_url_mappings', 'english.csv'), 'r') as f:
        i = 0
        for line in f.readlines():
            if i == 0:      # skip header
                i += 1
                continue
            else:
                if i < 3121850:
                    i += 1
                    continue
                bc, url = line.strip().split(',')
                bc_dir = os.path.join(output_dir, bc)

                # if i % 50 == 0:
                    # time.sleep(0.1)
                    # print 'bi_concept: {}; num_imgs: {}'.format(bc, i)
                i += 1

                # Make bi_concept directory if it doesn't exist
                if not os.path.exists(bc_dir):
                    os.makedirs(bc_dir)

                # Retrive image and save
                fn = os.path.basename(url)
                fp = os.path.join(bc_dir, fn)

                # Skip if file exists
                if os.path.exists(fp):
                    continue

                # Old sequential way:
                # retrieve_img_and_process([url, fp])

                urls.append(url)
                fps.append(fp)
    print 'done getting urls'

    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(100)
    urls_and_fps = zip(urls, fps)
    for i in range(0, len(urls), 1000):
        print i
        results = pool.map(retrieve_img_and_process, urls_and_fps[i:i+1000])

########################################################################################################################
# Sentibank + MVSO
########################################################################################################################

# Helper that just wraps dataset-specific functions
def get_bc2sent(dataset):
    if dataset == 'Sentibank':
        return get_Sentibank_bc2sent()
    elif dataset == 'MVSO':
        return get_MVSO_bc2sent()
    else:
        print 'unknown dataset: {}'.format(dataset)

def get_bc2emo(dataset):
    if dataset == 'Sentibank':
        return get_Sentibank_bc2emo()
    elif dataset == 'MVSO':
        return get_MVSO_bc2emo()
    else:
        print 'unknown dataset: {}'.format(dataset)

def get_bc2idx(dataset):
    if dataset == 'Sentibank':
        return get_Sentibank_bc2idx()
    elif dataset == 'MVSO':
        return get_MVSO_bc2idx()
    else:
        print 'unknown dataset: {}'.format(dataset)

# Writing images to tfrecords
def write_VSO_to_tfrecords(dataset, split=[0.8, 0.1, 0.1]):
    """Create tfrecord file for each biconcept for train,valid,test"""
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # Get lookups for each objective in order to label
    bc2sent = get_bc2sent(dataset)
    bc2emo = get_bc2emo(dataset)
    bc2idx = get_bc2idx(dataset)

    # Read config for sent_neutral_absval
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config = read_yaml(os.path.join(__location__, 'config.yaml'))
    sent_neutral_absval = config['sent_neutral_absval']

    # Iterate over biconcept folders
    if dataset == 'Sentibank':
         bc_path = SENTIBANK_BC_PATH
         dataset_path = SENTIBANK_FLICKR_PATH
    elif dataset == 'MVSO':
        bc_path = MVSO_BC_PATH
        dataset_path = MVSO_PATH
    else:
        print 'unknown dataset: {}'.format(dataset)

    for bc in [d for d in os.listdir(bc_path) if not d.startswith('.')]:
        print bc

        # Get filepaths of each image
        cur_bc_path = os.path.join(bc_path, bc)
        img_fns = [f for f in os.listdir(cur_bc_path) if f.endswith('jpg')]
        img_fps = [os.path.join(cur_bc_path, fn) for fn in img_fns]

        # Make directory for tfrecords - train, valid, test
        if not os.path.exists(os.path.join(dataset_path, 'tfrecords')):
            os.mkdir(os.path.join(dataset_path, 'tfrecords'))
        for name in ['train', 'valid', 'test']:
            if not os.path.exists(os.path.join(dataset_path, 'tfrecords', name)):
                os.mkdir(os.path.join(dataset_path, 'tfrecords', name))

        # Get tfrecord filepath and writer ready
        tfrecords_filename = '{}.tfrecords'.format(bc)
        tr_tfrecords_fp = os.path.join(dataset_path, 'tfrecords', 'train', tfrecords_filename)
        va_tfrecords_fp = os.path.join(dataset_path, 'tfrecords', 'valid', tfrecords_filename)
        te_tfrecords_fp = os.path.join(dataset_path, 'tfrecords', 'test', tfrecords_filename)
        tr_writer = tf.python_io.TFRecordWriter(tr_tfrecords_fp)
        va_writer = tf.python_io.TFRecordWriter(va_tfrecords_fp)
        te_writer = tf.python_io.TFRecordWriter(te_tfrecords_fp)
        train_endidx = int(split[0] * len(img_fps))
        valid_endidx = train_endidx + int(split[1] * len(img_fps))

        # Convert images to tfrecord Examples
        for i, img_fp in enumerate(img_fps):
            try:
                # Pull out image and labels and make example
                img = Image.open(img_fp)
                if img.mode != 'RGB' or img.format != 'JPEG':   # e.g. black and white (mode == 'L')
                    continue
                img = np.array(img)

                id = bc + '-' + os.path.basename(img_fp).split('.')[0]
                h, w = img.shape[0], img.shape[1]
                img_raw = img.tostring()
                # Can't use None as a feature, so just pass in a dummmy value. It'll be skipped anyway
                sent_reg_label = get_label(dataset, bc, 'sent_reg', bc_lookup=bc2sent)
                sent_reg_label = sent_reg_label if sent_reg_label else 0.0
                sent_biclass_label = get_label(dataset, bc, 'sent_biclass', bc_lookup=bc2sent, sent_neutral_absval=sent_neutral_absval)
                sent_biclass_label = sent_biclass_label if sent_biclass_label else 0
                sent_triclass_label = get_label(dataset, bc, 'sent_triclass', bc_lookup=bc2sent, sent_neutral_absval=sent_neutral_absval)
                sent_triclass_label = sent_triclass_label if sent_triclass_label else 0
                emo_label = get_label(dataset, bc, 'emo', bc_lookup=bc2emo)
                emo_label = emo_label if emo_label else 0
                bc_label = get_label(dataset, bc, 'bc', bc_lookup=bc2idx)
                bc_label = bc_label if bc_label else 0

                example = tf.train.Example(features=tf.train.Features(feature={
                    'id': _bytes_feature(id),
                    'h': _int64_feature(h),
                    'w': _int64_feature(w),
                    'img': _bytes_feature(img_raw),
                    'sent_reg': _float_feature(sent_reg_label),
                    'sent_biclass': _int64_feature(sent_biclass_label),
                    'sent_triclass': _int64_feature(sent_triclass_label),
                    'emo': _int64_feature(emo_label),
                    'bc': _int64_feature(bc_label)}))

                # Figure out which writer to use (train, valid, test)
                if i < train_endidx:
                    writer = tr_writer
                elif i >= train_endidx and i < valid_endidx:
                    writer = va_writer
                else:
                    writer = te_writer

                writer.write(example.SerializeToString())

            except Exception as e:
                print img_fp, e

    tr_writer.close()
    va_writer.close()
    te_writer.close()

def get_label(dataset, bc, obj, bc_lookup=None, sent_neutral_absval=None):
    """
    Return label from bi_concept string according to the objective (sentiment, emotion, biconcept)

    Handful of cases for sent where label doesn't exist. For example, candid_guy
    """
    if obj == 'sent_reg':
        if bc in bc_lookup:
            return bc_lookup[bc]
        else:
            return None
    elif obj == 'sent_biclass' or obj == 'sent_triclass':
        if bc in bc_lookup:
            return map_label_to_int(dataset, bc_lookup[bc], obj, sent_neutral_absval=sent_neutral_absval)
        else:
            return None
    elif obj == 'emo':
        if dataset == 'Sentibank':
            if len(bc_lookup[bc]) > 0:
                # TODO: what if there's a tie? (e.g. anger: 1, fear: 1) (this is probably pretty common)
                emo = bc_lookup[bc].most_common(1)[0][0]    # list of tuples of most occurring elements
                # print bc, emo, map_label_to_int(emo, obj)
                return map_label_to_int(dataset, emo, obj)
            else:       # no emotions for biconcept
                return None
        elif dataset == 'MVSO':
            if bc in bc_lookup:
                return map_label_to_int(dataset, bc_lookup[bc], obj)
            else:
                return None
        else:
            print 'unknown dataset: {}'.format(dataset)
    elif obj == 'bc':
        return map_label_to_int(dataset, bc, obj, bc2idx=bc_lookup)

def map_label_to_int(dataset, label, obj, sent_neutral_absval=None, bc2idx=None):
    """Map emo and bc string labels to int for classification tasks"""
    if obj == 'sent_biclass':
        label = 'neg' if label < 0 else 'pos'
        d = {'neg': 0, 'pos': 1}
        return SENT_BICLASS_LABEL2INT[label]
    elif obj == 'sent_triclass':
        if label > sent_neutral_absval:
            label = 'pos'
        elif label < -1 * sent_neutral_absval:
            label = 'neg'
        else:
            label = 'neutral'
        return SENT_TRICLASS_LABEL2INT[label]
    elif obj == 'emo':
        if dataset == 'Sentibank':
            return SENTIBANK_EMO_LABEL2INT[label]
        elif dataset == 'MVSO':
            return MVSO_EMO_LABEL2INT[label]
        else:
            print 'unknown dataset: {}'.format(dataset)
    elif obj == 'bc':
        return bc2idx[label]

def get_all_VSO_img_fps(dataset):
    """Return dictionary mapping bi_concept to list of img file paths"""
    path = SENTIBANK_BC_PATH if dataset == 'Sentibank' else os.path.join(MVSO_PATH, 'imgs')

    bc2img_fps = {}
    for bc in [d for d in os.listdir(path) if not d.startswith('.')]:
        cur_bc_path = os.path.join(path, bc)
        img_fns = [f for f in os.listdir(cur_bc_path) if f.endswith('jpg')]
        img_fps = [os.path.join(cur_bc_path, fn) for fn in img_fns]
        bc2img_fps[bc] = img_fps

    return bc2img_fps

def move_bad_jpgs(dataset):
    """Move bad jpegs out of biconcept folders using bad jpgs from remove_corrupted.lua"""
    if dataset == 'Sentibank':
         # Make directory to store bad jpgs
        bad_jpgs_dir = os.path.join(SENTIBANK_FLICKR_PATH, 'bad_jpgs')
        if not os.path.exists(bad_jpgs_dir):
            os.mkdir(bad_jpgs_dir)

        bad_jpg_fns = open(os.path.join(SENTIBANK_FLICKR_PATH, 'bad_imgs.txt'), 'r').readlines()
        bad_jpg_fns = [f.strip('\n') for f in bad_jpg_fns]

        bc2img_fps = get_all_VSO_img_fps(dataset)
        for bc, img_fps in bc2img_fps.items():
            print bc
            for img_fp in img_fps:
                img_fn = os.path.basename(img_fp)
                if img_fn in bad_jpg_fns:
                    print bc, img_fn
                    os.rename(img_fp, os.path.join(bad_jpgs_dir, '{}-{}'.format(bc, img_fn)))
    elif dataset == 'MVSO':
        print 'MVSO not impelemented yet'
    else:
        print 'unknown dataset: {}'.format(dataset)

def save_bc_channel_mean_std(dataset):
    """Save channel-wise mean and stddev so we can standardize"""
    # Make directory to save
    if dataset == 'Sentibank':
        out_dir = os.path.join(SENTIBANK_FLICKR_PATH, 'bc_channelmeanstd')
    else:
        out_dir = os.path.join(MVSO_PATH, 'bc_channelmeanstd')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    bc2mean = {}
    bc2std = {}

    bc2img_fps = get_all_VSO_img_fps(dataset)
    for bc, img_fps in bc2img_fps.items():
        mean, std = np.zeros(3), np.zeros(3)
        n = 0
        for img_fp in img_fps:
            try:
                im = Image.open(img_fp)
                if im.mode != 'RGB':      # type L, P, etc. shows some type of Flickr unavailable photo img
                    os.remove(img_fp)
                    continue
                im = np.array(im)
                im = im.astype(np.float32)
                im /= 256.0               # convert to [0,)
                for c in range(3):
                    mean[c] += im[:,:,c].mean()
                    std[c] += im[:,:,c].std()
                n += 1
            except Exception as e:
                print e

        mean /= float(n)
        std /= float(n)
        print '{} mean: {}'.format(bc, mean)
        print '{} std: {}'.format(bc, std)

        bc2mean[bc] = mean
        bc2std[bc] = std

    with open(os.path.join(out_dir, 'bc2channelmean.pkl'.format(bc)), 'w') as f:
        pickle.dump(bc2mean, f, protocol=2)
    with open(os.path.join(out_dir, 'bc2channelstd.pkl'.format(bc)), 'w') as f:
        pickle.dump(bc2std, f, protocol=2)

########################################################################################################################
# You image emotion
########################################################################################################################
def _get_you_imemo_urls():
    """
    Return URLs for images where majority of 5 (most of the time it's 5) AMT reviewers agreed with the emotion label

    Stats
    -----
    Total 23166
    excitement 2918
    sadness 2902
    contentment 5356
    disgust 1650
    anger 1255
    awe 3133
    fear 1029
    amusement 4923
    """
    csv_fns = [f for f in os.listdir(YOU_IMEMO_PATH) if f.endswith('csv')]
    emo2urls = defaultdict(list)
    for fn in csv_fns:
        emo = fn.split('_')[0]
        fp = os.path.join(YOU_IMEMO_PATH, fn)
        with open(fp) as f:
            for line in f.readlines():
                _, url, disagree, agree = line.split(',')
                if int(agree) > int(disagree):
                    emo2urls[emo].append(url)
    return emo2urls

def retrieve_you_imemo_imgs(out_dir=os.path.join(YOU_IMEMO_PATH, 'imgs')):
    """Download images for each emotion for You im_emo dataset"""
    emo2urls = _get_you_imemo_urls()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for emo, urls in emo2urls.items():
        print emo
        if emo != 'fear' and emo != 'amusement':
            continue
        emo_dir = os.path.join(out_dir, emo)
        if not os.path.exists(emo_dir):
            os.makedirs(emo_dir)
        for url in urls:
            img_name = os.path.basename(url)
            urllib.urlretrieve(url, os.path.join(emo_dir, img_name))

########################################################################################################################
# Plutchik's wheel of emotions and color
########################################################################################################################
def save_plutchik_color_imgs():
    """Parse txt file with emotions and RGB colors, save solid color images"""
    # label2rgb = {}
    with open(os.path.join(PLUTCHIK_PATH, 'plutchik_colors.txt'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            m = re.match(r'(\w+) - R: (\w+) G: (\w+) B: (\w+)', line)
            label, r, g, b = m.group(1), m.group(2), m.group(3), m.group(4)
            print i, line, r, g, b
            # label2rgb[label] = [r,g,b]
            im = np.zeros([256, 256, 3], 'uint8')
            im[:,:,0] = np.ones([256, 256]) * int(r)
            im[:,:,1] = np.ones([256, 256]) * int(g)
            im[:,:,2] = np.ones([256, 256]) * int(b)

            import scipy.misc
            scipy.misc.imsave(os.path.join(PLUTCHIK_PATH, '{}_{}.jpg'.format(i, label)), im)

########################################################################################################################
########################################################################################################################
########################################################################################################################
#
# Videos
#
########################################################################################################################
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Frames
########################################################################################################################
def save_video_frames(vids_dir):
    """
    Loop over subdirs within vids_dir and save frames to subdir/frames/

    Parameters
    ----------
    vids_dir: directory within VIDEOS_PATH that contains sub-directories, each which may contain a movie
    """

    # vid_exts = ['mp4', 'avi']
    mr = MovieReader()

    vids_path = os.path.join(VIDEOS_PATH, vids_dir)
    ext2count = defaultdict(int)
    i = 0
    successes = []
    for vid_name in [d for d in os.listdir(vids_path) if not d.startswith('.')]:
        # Skip if frames/ already exists and has some frames in it
        if os.path.exists(os.path.join(vids_path, vid_name, 'frames')):
            if len(os.listdir(os.path.join(vids_path, vid_name, 'frames'))) != 0:
                continue

        # Get the actual video file, while also removing any sample video files if they are there
        vid_dirpath = os.path.join(vids_path, vid_name)
        files = os.listdir(vid_dirpath)
        movie_file = None
        vid_ext = None
        for f in files:
            if 'sample' in f.lower():
                try:
                    os.remove(os.path.join(vid_dirpath, f))
                except Exception as e:
                    print 'Removed sample file {} for {}'.format(f, vid_name)
        for f in files:
            for ext in VID_EXTS:
                if f.endswith(ext):
                    movie_file = f
                    ext2count[ext] += 1
                    vid_ext = ext

        # Try to save frames for video file
        if movie_file:
            print '=' * 100
            print 'Video: {}'.format(vid_name)
            print 'Format: {}'.format(vid_ext)
            movie_path = os.path.join(vid_dirpath, movie_file)
            try:
                mr.write_frames(movie_path)
                i += 1
                successes.append(vid_name)
                # TODO: should check number of frames -- sometimes only a few saved, there's an error going through file
            except Exception as e:
                print e

    print '=' * 100
    print 'Created frames for {}'.format(successes)
    print 'Extension counts: {}'.format(ext2count)      # will only be for movies without frames/
    print 'Created frames for {} videos'.format(i)

def convert_avis_to_mp4s(vids_dir):
    """
    Convert videos from avi to mp4

    Notes
    -----
    Frames cannot be extracted from avi files using MovieReader currently, thus convert to more standard mp4 format

    Parameters
    ----------
    vids_dir: directory within VIDEOS_PATH that contains sub-directories, each which may contain a movie
    """
    vids_path = os.path.join(VIDEOS_PATH, vids_dir)
    vid_dirs = [d for d in os.listdir(vids_path) if not d.startswith('.')]
    for vid_dir in vid_dirs:
        # Find avi file if it exists
        vid_dirpath = os.path.join(vids_path, vid_dir)
        filenames = os.listdir(vid_dirpath)
        avi_fn = None
        for fn in filenames:
            if fn.endswith('avi'):
                avi_fn = fn

        # Convert to mp4, clean up if it succeeds
        if avi_fn:
            try:
                print '=' * 100
                print '=' * 100
                print '=' * 100
                print 'Found avi file to convert for: {}'.format(vid_dir)
                mp4_fn = avi_fn.split('.avi')[0] + '.mp4'
                avi_path = os.path.join(vid_dirpath, avi_fn)
                mp4_path = os.path.join(vid_dirpath, mp4_fn)
                bash_command = ['avconv', '-i'] + [avi_path] + ['-c:v', 'libx264', '-c:a', 'copy'] + [mp4_path]
                # Not using split() on bash command string because path's may have spaces
                print bash_command
                subprocess.call(bash_command, stdout=subprocess.PIPE)
                print 'Done converting, will remove avi file'
                os.remove(avi_path)
            except Exception as e:
                print e

def save_credits_index(vids_dir, overwrite_files=False):
    """
    Save index of frames/ for when credits start
    """
    cl = CreditsLocator(overwrite_files=overwrite_files)
    vids_path = os.path.join(VIDEOS_PATH, vids_dir)
    vid_dirs = [d for d in os.listdir(vids_path) if not d.startswith('.')]
    not_located = []
    for vid_dir in vid_dirs:
        print '=' * 100
        print vid_dir
        located = cl.locate_credits(os.path.join(vids_path, vid_dir))
        if not located:
            not_located.append(vid_dir)

    print '=' * 100
    print 'Credits not located for {} movies:'.format(len(not_located))
    pprint(sorted(not_located))

########################################################################################################################
# VideoPath DB
########################################################################################################################
def create_videopath_db():
    """
    Create sqllite db storing information about video paths, formats, frames, etc.
    """
    def get_movie_fn_if_exists(files):
        for f in files:
            if 'sample' in f.lower():
                continue
            for ext in VID_EXTS:
                if f.endswith(ext):
                    return f
        return None

    def get_dataset_name_from_dir(dir):
        # MovieQA_full_movies -> MovieQA
        if dir == 'MovieQA_full_movies':
            return 'MovieQA'
        elif dir == 'M-VAD_full_movies':
            return 'M-VAD'
        else:
            return dir

    # Delete and recreate database
    if os.path.exists(VIDEOPATH_DB):
        os.remove(VIDEOPATH_DB)
    conn = sqlite3.connect(VIDEOPATH_DB)
    conn.execute('CREATE TABLE VideoPath('
                 'category TEXT,'
                 'title TEXT,'
                 'datasets TEXT,'
                 'dirpath TEXT,'
                 'movie_fn TEXT,'
                 'ext TEXT,'
                 'has_frames INTEGER,'
                 'num_frames INTEGER)')

    # Find all directories with movies
    with conn:
        cur = conn.cursor()
        for root, dirs, files in os.walk(VIDEOS_PATH):
            movie_fn = get_movie_fn_if_exists(files)
            if movie_fn:
                # root: data/videos/films/MovieQA_full_movies/Yes Man (2008)
                title = os.path.basename(root)
                category = root.split(VIDEOS_PATH)[1].split('/')[1]
                dirpath = root
                # print root
                datasets = get_dataset_name_from_dir(root.split(category)[1].split('/')[1])
                ext = movie_fn.split('.')[-1]
                has_frames = int(('frames' in dirs) and (len(os.listdir(os.path.join(root, 'frames'))) > 0))
                num_frames = len(os.listdir(os.path.join(root, 'frames'))) if has_frames else 0

                print category, title, datasets, dirpath, movie_fn, ext, has_frames, num_frames

                cur.execute("INSERT INTO VideoPath VALUES(?, ?, ?, ?, ?, ?, ?, ?)", (
                    category,
                    title.decode('utf8'),
                    datasets,
                    dirpath.decode('utf8'),
                    movie_fn.decode('utf8'),
                    ext,
                    has_frames,
                    num_frames
                ))

    # TODO and note on datasets field (not high priority, as datasets not being used right now)
    # 1) datasets is meant to track which existing datasets movie is also a part of.
    # For instance, the bulk of the movies are from the MovieQA and M-VAD datasets.
    # These were chosen in the case we wanted to do further analysis on these movies -- these datasets
    # provide extra DVS descriptions and metadata. (They also include large chunks of the movie, but not the full
    # movie, which is why they were downloaded in full, and hence the name of the directories, e.g. MovieQA_full_movies)
    # 2) However, there is some overlap between these datasets (as well as other datasets, such as CMU_movie_tropes).
    # datasets is supposed to be a comma-separated list of these datasets. In order to make it complete, I should
    # match the movies from the txt files containing the list of movies and upsert into the table.

########################################################################################################################
# VideoMetadata DB
########################################################################################################################
def match_film_metadata():
    """
    Get metadata for each film using CMU_movie_summary dataset

    Notes
    -----
    Assumes each title is similar to the format: <title> (<year>), e.g. King's Speech (2010)

    movie.metadata.tsv columns:
    # 1. Wikipedia movie ID ('975900')
    # 2. Freebase movie ID ('/m/03vyhn')
    # 3. Movie name ('Ghosts of Mars')
    # 4. Movie release date (2001-08-24)
    # 5. Movie box office revenue  ('14010832')
    # 6. Movie runtime ('98.0')
    # 7. Movie languages (Freebase ID:name tuples) ('{"/m/02h40lc": "English Language"}')
    # 8. Movie countries (Freebase ID:name tuples) ('{"/m/09c7w0": "United States of America"}'
    # 9. Movie genres (Freebase ID:name tuples) ('{"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction", ...}\n'}
    """

    # Get all metadata
    movie2metadata = {}
    with open(os.path.join(CMU_PATH, 'movie.metadata.tsv'), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            movie2metadata[line[2]] = {'date': line[3],
                                       'revenue': None if line[4] == '' else int(line[4]),
                                       'runtime': None if line[5] == '' else float(line[5]),
                                       'genres': json.loads(line[8]).values()}
    movies = set(movie2metadata.keys())

    # Get tropes for each movie in tvtropes data
    movie2tropes = defaultdict(list)
    with open(os.path.join(CMU_PATH, 'tvtropes.clusters.txt'), 'r') as f:
        for line in f.readlines():
            trope, movie_data = line.split('\t')
            movie_data = json.loads(movie_data)     # char, movie, id, actor
            movie2tropes[movie_data['movie']].append(trope)

    # Write moves to file for the hell of it, so I can spot check
    with io.open('notes/CMU_tvtropes_movies.txt', 'w') as f:
        for movie in movie2tropes.keys():
            f.write(movie)
            f.write(u'\n')

    # Match videos to movies with metadata
    # Most are a good match, and there's only around 500 movies, so I manually cleaned it as follows:
    # I ran it once and saved stdout to a text file and just went through all the ones that aren't
    # This was done when M-VAD and MovieQA were added to MoviePath DB,
    #   and a few movies from other, CMU_tv_tropes, and animated
    # Then I just found all the ones that were incorrect or not there.
    # The biggest mismatches are for movies with sequels
    # The ones not there are mostly newer movies
    # The following was done manually
    manually_matched = {
        'The Hobbit The Desolation of Smaug': None,
        'Wanted': 'Wanted', # not Wanted 2
        'Cruel Intentions': 'Cruel Intentions',
        'Blue Jasmine': None,
        'Still Alice': None,
        'The Godfather Part I': 'The Godfather',
        'Twilight Saga': 'Twilight',
        'A Royal Night Out': None,
        'American Heist': None,
        'Gone Girl': None,
        'A Walk Among the Tombstones': None,
        '12 Years a Slave': None,
        'Avatar': 'Avatar',
        'Wild Things': 'Wild Things',
        'Oceans 11': "Ocean's Eleven",
        'Chappie': None,
        'Kung Fu Panda': 'Kung Fu Panda',
        'Her': 'Her',
        'X-Men Days of Future Past': None,
        'How to Train Your Dragon 2': None,
        'Carol': 'Carol',
        'The Intouchables': 'Intouchables',
        '22 Jump Street': None,
        'Marley and Me': None,
        'Before I Go to Sleep': None,
        'Taken': 'Taken',
        '2 Guns': None,
        '3 Days to Kill': None,
        'The Butterfly Effect': 'The Butterfly Effect',
        'Short Term 12': None,
        'Elizabeth': 'Elizabeth',
        'American Psycho': 'American Psycho',
        'Men In Black': 'Men In Black',
        'This Is 40': None,
        'The Grand Budapest Hotel': None,
        'Zipper': None,
        'Mrs Doubtfire': 'Mrs. Doubtfire',
        'The Godfather Part 3': 'The Godfather Part III',
        'Bad Santa': 'Bad Santa',
        'Divergent': None,
        'The Hobbit The Battle of The Five Armies': None,
        'Cold in July': None,
        'Absolutely Anything': None,
        'Harry Potter And The Deathly Hallows Part 2': 'Harry Potter and the Deathly Hallows \xe2\x80\x93 Part 2',
        'A Walk in the Woods': None,
        'Back to the Future II': 'Back to the Future Part II',
        'I Robot': 'I, Robot',
        'About Time': None,
        '71': None,
        'X2 X-Men United': None,
        'Iron Man': 'Iron Man',
        'Captain America Civil War': None,
        'Shrek': 'Shrek',
        'Zootopia': None,
        'Big Hero 6': None,
        'The Wind Rises': None,
        'Bruno': 'Br\xc3\xbcno',
        'The Guilt Trip': None,
        'The Adventures of Tintin': None,

    }

    result = {}

    conn = sqlite3.connect(VIDEOPATH_DB)
    with conn:
        cur = conn.cursor()
        rows = cur.execute("SELECT title FROM VideoPath WHERE category=='films'")
        for row in rows:
            title = row[0]
            title = title.encode('utf-8')
            m = re.match(r'(.+)\(\d+\)?$', title)
            title = m.group(1)

            if title in manually_matched:
                match = manually_matched[title]
                if match:
                    result[title] = movie2metadata[match]
                    continue
            else:
                matched = sorted([(fuzz.ratio(title, movie_name), movie_name) for movie_name in movies])
                matched = [t for s, t in matched[::-1][:10]]        # top 10
                match = matched[0]
                result[title] = movie2metadata[match]

                # print title, matched
    return result

def get_shorts_metadata():
    """
    Aggregate metadata for each video in data/videos/shorts

    Notes
    -----
    Each folder downloaded from Vimeo should have 1) <title>.info.json, 2) <title>.annotations.xml, and 3) <title>.description

    Returns
    -------
    short2metadata: dict
        - key is title (name of dir) This title is the common key with VIDEOPATH_DB
        - value is dictionary of metadata
    - The metadata dictionary is a subset of the info.json data and includes the description. For all the videos I
    spotchecked, annotatiosn is empty, so I'm skipping it for now.
    """
    short2metadata = {}

    def parse_info_file(fp):
        data = json.load(open(fp, 'r'))
        info = {
            'comment_count': data.get('comment_count'),
            'description': data.get('description'),
            'display_id': data.get('display_id'),  # not sure what's the diff with 'id' -- one video I checked they were the same
            'duration': data.get('duration'),
            'fps': data.get('fps'),
            'fulltitle': data.get('fulltitle'),
            'id': data.get('id'),
            'like_count': data.get('like_count'),
            'subtitles': data.get('subtitles'),
            'title': data.get('title'),
            'uploader': data.get('uploader'),
            'uploader_id': data.get('uploader_id'),
            'uploader_url': data.get('uploader_url'),
            'upload_date': data.get('upload_date'),
            'view_count': data.get('view_count')
        }
        return info

    conn = sqlite3.connect(VIDEOPATH_DB)
    with conn:
        cur = conn.cursor()
        rows = cur.execute("SELECT dirpath, title FROM VideoPath WHERE category=='shorts'")
        for row in rows:
            dirpath, title = row[0], row[1]
            # print title
            info_fp = os.path.join(dirpath, title +  u'.info.json')
            info = parse_info_file(info_fp)
            short2metadata[title] = info
    return short2metadata

def create_videometadata_db():
    """
    Create VideoMetadata DB -- right now it's just a pkl file (I was looking for a lightweight no SQL database.)
    The keys are titles (common key with VideoPath DB).
    """
    db = {}

    print 'Getting shorts metadata'
    short2metadata = get_shorts_metadata()
    for title, metadata in short2metadata.items():
        db[title] = metadata

    print 'Getting films metadata'
    film2metadata = match_film_metadata()
    for title, metadata in film2metadata.items():
        db[title] = metadata

    print 'Saving'
    with open(VIDEOMETADATA_DB, 'w') as f:
        pickle.dump(db, f, protocol=2)

    print 'Done'


if __name__ == '__main__':

    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='Download and process data')
    parser.add_argument('--MVSO_dl_imgs', dest='MVSO_dl_imgs', action='store_true')
    parser.add_argument('--MVSO_bc2emo2val', dest='MVSO_bc2emo2val', action='store_true')
    parser.add_argument('--VSO_dataset', dest='VSO_dataset', default='Sentibank', help='Sentibank,MVSO')
    parser.add_argument('--VSO_img_fps', dest='VSO_img_fps', action='store_true')
    parser.add_argument('--bc2sent', dest='bc2sent', action='store_true')
    parser.add_argument('--bc2emo', dest='bc2emo', action='store_true')
    parser.add_argument('--bc2idx', dest='bc2idx', action='store_true')
    parser.add_argument('--VSO_to_tfrecords', dest='VSO_to_tfrecords', action='store_true')
    parser.add_argument('--move_bad_jpgs', dest='move_bad_jpgs', action='store_true')
    parser.add_argument('--bc_channel_mean_std', dest='bc_channel_mean_std', action='store_true')
    parser.add_argument('--you_dl_imgs', dest='you_dl_imgs', action='store_true')
    parser.add_argument('--save_plutchik_color_imgs', dest='save_plutchik_color_imgs', action='store_true')
    parser.add_argument('--save_video_frames', dest='save_video_frames', action='store_true')
    parser.add_argument('--convert_avis_to_mp4s', dest='convert_avis_to_mp4s', action='store_true')
    parser.add_argument('--save_credits_index', dest='save_credits_index', action='store_true')
    parser.add_argument('--save_credits_index_overwrite', dest='save_credits_index_overwrite', default=False,
                        action='store_true', help='overwrite credits_index.txt files')
    parser.add_argument('--create_videopath_db', dest='create_videopath_db', action='store_true')
    parser.add_argument('--match_film_metadata', dest='match_film_metadata', action='store_true')
    parser.add_argument('--get_shorts_metadata', dest='get_shorts_metadata', action='store_true')
    parser.add_argument('--create_videometadata_db', dest='create_videometadata_db', action='store_true')
    parser.add_argument('--vids_dir', dest='vids_dir', default=None,
                        help='folder that contains dirs (one movie each), e.g. films/MovieQA_full_movies')

    cmdline = parser.parse_args()

    if cmdline.MVSO_dl_imgs:
        download_MVSO_imgs()
    elif cmdline.MVSO_bc2emo2val:
        pprint(get_MVSO_bc2emo2val())
    elif cmdline.VSO_img_fps:
        bc2img_fps = get_all_VSO_img_fps(cmdline.VSO_dataset)
        print len([k for k, v in bc2img_fps.items() if len(v) > 120])
        print len(bc2img_fps)
    elif cmdline.bc2sent:
        pprint(get_bc2sent(cmdline.VSO_dataset))
    elif cmdline.bc2emo:
        pprint(get_bc2emo(cmdline.VSO_dataset))
    elif cmdline.bc2idx:
        pprint(get_bc2idx(cmdline.VSO_dataset))
    elif cmdline.VSO_to_tfrecords:
        write_VSO_to_tfrecords(cmdline.VSO_dataset)
    elif cmdline.move_bad_jpgs:
        move_bad_jpgs(cmdline.VSO_dataset)
    elif cmdline.bc_channel_mean_std:
        save_bc_channel_mean_std(cmdline.VSO_dataset)
    elif cmdline.you_dl_imgs:
        retrieve_you_imemo_imgs()
    elif cmdline.save_plutchik_color_imgs:
        save_plutchik_color_imgs()
    elif cmdline.save_video_frames:
        save_video_frames(cmdline.vids_dir)
    elif cmdline.convert_avis_to_mp4s:
        convert_avis_to_mp4s(cmdline.vids_dir)
    elif cmdline.save_credits_index:
        save_credits_index(cmdline.vids_dir, overwrite_files=cmdline.save_credits_index_overwrite)
    elif cmdline.create_videopath_db:
        create_videopath_db()
    elif cmdline.match_film_metadata:
        pprint(match_film_metadata(cmdline.vids_dir))
    elif cmdline.get_shorts_metadata:
        pprint(get_shorts_metadata())
    elif cmdline.create_videometadata_db:
        create_videometadata_db()