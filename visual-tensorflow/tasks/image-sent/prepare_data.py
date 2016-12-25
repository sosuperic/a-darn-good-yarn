# Prepare the data (parse, convert to tfrecords, download, etc.)

import argparse
from collections import defaultdict, Counter
import numpy as np
import os
import pickle
from PIL import Image
from pprint import pprint
import re
from skimage import color
import tensorflow as tf
import urllib

from core.utils.utils import read_yaml
from core.utils.MovieReader import MovieReader

# Sentibank - bi_concepts1553: mapping ajdective noun pairs to sentiment
FLICKR_PATH = 'data/Sentibank/Flickr/'
BC_PATH = 'data/Sentibank/Flickr/bi_concepts1553'
BC_TRAINTEST_PATH = 'data/Sentibank/Flickr/trainingAndTestingPartition'
EMOLEX_PATH = 'data/emolex/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'

# You dataset - 20k images with emotions
YOU_IMEMO_PATH = 'data/you_imemo/agg'

# MVSO dataset - mutlilingual, larger version of Sentibank VSO; also has emotions
MVSO_PATH = 'data/MVSO'

########################################################################################################################
# Sentibank
########################################################################################################################

# Getting data structures for each objective
def get_bc2sent():
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

def get_bc2emo():
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
    bc2img_fps = get_all_bc_img_fps()
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

def get_all_bc_img_fps():
    """Return dictionary mapping bi_concept to list of img file paths"""
    bc2img_fps = {}
    for bc in [d for d in os.listdir(BC_PATH) if not d.startswith('.')]:
        cur_bc_path = os.path.join(BC_PATH, bc)
        img_fns = [f for f in os.listdir(cur_bc_path) if f.endswith('jpg')]
        img_fps = [os.path.join(cur_bc_path, fn) for fn in img_fns]
        bc2img_fps[bc] = img_fps

    return bc2img_fps

def get_bc2idx():
    """Return dictionary mapping biconcept to idx"""
    bc2idx = {}
    for i, bc in enumerate([d for d in os.listdir(BC_PATH) if not d.startswith('.')]):
        bc2idx[bc] = i
    return bc2idx

def get_all_bc_img_fps():
    """Return dictionary mapping bi_concept to list of img file paths"""
    bc2img_fps = {}
    for bc in [d for d in os.listdir(BC_PATH) if not d.startswith('.')]:
        cur_bc_path = os.path.join(BC_PATH, bc)
        img_fns = [f for f in os.listdir(cur_bc_path) if f.endswith('jpg')]
        img_fps = [os.path.join(cur_bc_path, fn) for fn in img_fns]
        bc2img_fps[bc] = img_fps

    return bc2img_fps

# Parsing originally provided test train partition
def get_bc_traintest():
    """Return and save dictionary mapping each bc to train/test to positive/negative to path"""
    bc2split2posneg2path = defaultdict(dict)
    sets = ['reduced{}'.format(i) for i in range(1, 6)] + ['fullset']
    for s in sets:
        print s
        with open(os.path.join(BC_TRAINTEST_PATH, 'trainingAndTesting_{}'.format(s)), 'r') as f:
            cur_bc, cur_split = '', ''
            for line in f.readlines():
                line = line.split()
                if len(line) == 3:
                    cur_bc, cur_split = line[0], line[1]
                if len(line) == 1:
                    line = line[0]
                    pos_or_neg = 'pos' if line.startswith('+') else 'neg'
                    relative_path = line.split(':')[1].replace('\\', '/')
                    full_path = os.path.join(BC_PATH, relative_path)
                    if cur_split in bc2split2posneg2path[cur_bc]:
                        if pos_or_neg in bc2split2posneg2path[cur_bc][cur_split]:
                            bc2split2posneg2path[cur_bc][cur_split][pos_or_neg].append(full_path)
                        else:
                            bc2split2posneg2path[cur_bc][cur_split][pos_or_neg] = [full_path]
                    else:
                        bc2split2posneg2path[cur_bc][cur_split] = {}
                        bc2split2posneg2path[cur_bc][cur_split][pos_or_neg] = [full_path]

        f = open(os.path.join(BC_TRAINTEST_PATH, 'bc2split2posneg2path_{}.pkl'.format(s)), 'w')
        pickle.dump(bc2split2posneg2path, f, protocol=2)
        f.close()

    return bc2split2posneg2path

# Writing images to tfrecords
def write_sentibank_to_tfrecords(split=[0.8, 0.1, 0.1]):
    """Create tfrecord file for each biconcept for train,valid,test"""
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # Get lookups for each objective in order to label
    bc2sent = get_bc2sent()
    bc2emo = get_bc2emo()
    bc2idx = get_bc2idx()

    # Read config for sent_neutral_absval
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config = read_yaml(os.path.join(__location__, 'config.yaml'))
    sent_neutral_absval = config['sent_neutral_absval']

    # Iterate over biconcept folders
    for bc in [d for d in os.listdir(BC_PATH) if not d.startswith('.')]:
        print bc

        # Get filepaths of each image
        cur_bc_path = os.path.join(BC_PATH, bc)
        img_fns = [f for f in os.listdir(cur_bc_path) if f.endswith('jpg')]
        img_fps = [os.path.join(cur_bc_path, fn) for fn in img_fns]

        # Make directory for tfrecords - train, valid, test
        if not os.path.exists(os.path.join(FLICKR_PATH, 'tfrecords')):
            os.mkdir(os.path.join(FLICKR_PATH, 'tfrecords'))
        for name in ['train', 'valid', 'test']:
            if not os.path.exists(os.path.join(FLICKR_PATH, 'tfrecords', name)):
                os.mkdir(os.path.join(FLICKR_PATH, 'tfrecords', name))

        # Get tfrecord filepath and writer ready
        tfrecords_filename = '{}.tfrecords'.format(bc)
        tr_tfrecords_fp = os.path.join(FLICKR_PATH, 'tfrecords', 'train', tfrecords_filename)
        va_tfrecords_fp = os.path.join(FLICKR_PATH, 'tfrecords', 'valid', tfrecords_filename)
        te_tfrecords_fp = os.path.join(FLICKR_PATH, 'tfrecords', 'test', tfrecords_filename)
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

                # print 'a'
                h, w = img.shape[0], img.shape[1]
                # print 'b'
                img_raw = img.tostring()
                # print 'c'
                # Can't use None as a feature, so just pass in a dummmy value. It'll be skipped anyway
                sent_reg_label = get_label(bc, 'sent_reg', bc_lookup=bc2sent)
                sent_reg_label = sent_reg_label if sent_reg_label else 0.0
                # print 'd', sent_reg_label
                sent_biclass_label = get_label(bc, 'sent_biclass', bc_lookup=bc2sent, sent_neutral_absval=sent_neutral_absval)
                sent_biclass_label = sent_biclass_label if sent_biclass_label else 0
                # print 'e', sent_biclass_label
                sent_triclass_label = get_label(bc, 'sent_triclass', bc_lookup=bc2sent, sent_neutral_absval=sent_neutral_absval)
                sent_triclass_label = sent_triclass_label if sent_triclass_label else 0
                # print 'f', sent_triclass_label
                emo_label = get_label(bc, 'emo', bc_lookup=bc2emo)
                emo_label = emo_label if emo_label else 0
                # print 'g', emo_label
                bc_label = get_label(bc, 'bc', bc_lookup=bc2idx)
                bc_label = bc_label if bc_label else 0
                # print 'h', bc_label

                example = tf.train.Example(features=tf.train.Features(feature={
                    'h': _int64_feature(h),
                    'w': _int64_feature(w),
                    'img': _bytes_feature(img_raw),
                    'sent_reg': _float_feature(sent_reg_label),
                    'sent_biclass': _int64_feature(sent_biclass_label),
                    'sent_triclass': _int64_feature(sent_triclass_label),
                    'emo': _int64_feature(emo_label),
                    'bc': _int64_feature(bc_label)}))


                # print 'i'

                # Figure out which writer to use (train, valid, test)
                if i < train_endidx:
                    writer = tr_writer
                elif i >= train_endidx and i < valid_endidx:
                    writer = va_writer
                else:
                    writer = te_writer

                # print 'j'

                writer.write(example.SerializeToString())

                # print '@@@@k@@@@'

            except Exception as e:
                print img_fp, e
                # import sys
                # sys.exit()

        # break

    tr_writer.close()
    va_writer.close()
    te_writer.close()

# Get labels
def map_label_to_int(label, obj, sent_neutral_absval=None, bc2idx=None):
    """Map emo and bc string labels to int for classification tasks"""
    if obj == 'sent_biclass':
        label = 'neg' if label < 0 else 'pos'
        d = {'neg': 0, 'pos': 1}
        return d[label]
    elif obj == 'sent_triclass':
        if label > sent_neutral_absval:
            label = 'pos'
        elif label < -1 * sent_neutral_absval:
            label = 'neg'
        else:
            label = 'neutral'
        d = {'neg':0, 'neutral':1, 'pos':2}
        return d[label]
    elif obj == 'emo':
        d = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3,
             'joy': 4, 'sadness': 5, 'surprise': 6, 'trust': 7}
        return d[label]
    elif obj == 'bc':
        return bc2idx[label]


def get_label(bc, obj, bc_lookup=None, sent_neutral_absval=None):
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
            return map_label_to_int(bc_lookup[bc], obj, sent_neutral_absval=sent_neutral_absval)
        else:
            return None
    elif obj == 'emo':
        if len(bc_lookup[bc]) > 0:
            # TODO: what if there's a tie? (e.g. anger: 1, fear: 1) (this is probably pretty common)
            emo = bc_lookup[bc].most_common(1)[0][0]    # list of tuples of most occurring elements
            print bc, emo, map_label_to_int(emo, obj)
            return map_label_to_int(emo, obj)
        else:       # no emotions for biconcept
            return None
    elif obj == 'bc':
        return map_label_to_int(bc, obj, bc2idx=bc_lookup)

# Move bad jpegs out of folder
# Uses bad jpgs from remove_corrupted.lua
def move_bad_jpgs():
    """Move bad jpegs out of biconcept folders"""
    # Make directory to store bad jpgs
    bad_jpgs_dir = os.path.join(FLICKR_PATH, 'bad_jpgs')
    if not os.path.exists(bad_jpgs_dir):
        os.mkdir(bad_jpgs_dir)

    bad_jpg_fns = open(os.path.join(FLICKR_PATH, 'bad_imgs.txt'), 'r').readlines()
    bad_jpg_fns = [f.strip('\n') for f in bad_jpg_fns]

    bc2img_fps = get_all_bc_img_fps()
    for bc, img_fps in bc2img_fps.items():
        print bc
        for img_fp in img_fps:
            img_fn = os.path.basename(img_fp)
            if img_fn in bad_jpg_fns:
                print bc, img_fn
                os.rename(img_fp, os.path.join(bad_jpgs_dir, '{}-{}'.format(bc, img_fn)))

# Uses output from bash/identify_bad_jpegs.sh
# def move_bad_jpgs():
#     """Move bad jpegs out of biconcept folders"""
#     # Make directory to store bad jpgs
#     bad_jpgs_dir = os.path.join(FLICKR_PATH, 'bad_jpgs')
#     if not os.path.exists(bad_jpgs_dir):
#         os.mkdir(bad_jpgs_dir)
#
#     # For each bc folder, read the good jpgs list and find out the bad jpgs
#     jpg_check_dir = os.path.join(FLICKR_PATH, 'jpg_check')
#     for bc in os.listdir(jpg_check_dir):
#         with open(os.path.join(FLICKR_PATH, bc, 'ok_jpgs.txt'), 'r') as f:
#             good_fns = f.readlines()
#         all_jpg_fns = [f for f in os.listdir(os.path.join(BC_PATH, bc)) if f.endswith('txt')]
#         bad_fns = set(all_jpg_fns) - set(good_fns)
#
#         # Move bad jpg to bad jpg folder
#         for bad_fn in bad_fns:
#             if not os.path.exists(os.path.join(bad_jpgs_dir, bc)):
#                 os.mkdir(os.path.join(bad_jpgs_dir, bc))
#             os.rename(os.path.join(BC_PATH, bc, bad_fn), os.path.join(bad_jpgs_dir, bc, '{}-{}'.format(bc, bad_fn)))

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
# MVSO
########################################################################################################################
def get_MVSO_bc2sentiment():
    """Return dict from bi_concept to sentiment value"""
    bc2sentiment = {}
    with open(os.path.join(MVSO_PATH, 'mvso_sentiment', 'english.csv'), 'r') as f:
        for line in f.readlines():
            bc, sentiment = line.strip().split(',')
            bc2sentiment[bc] = float(sentiment)
    return bc2sentiment

def get_MVSO_bc2emotion2value():
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

def download_MVSO_imgs(output_dir=os.path.join(MVSO_PATH, 'imgs'), target_w=256, target_h=256):
    """Download, resize, and center crop images"""
    import time
    mr = MovieReader()          # used to resize and center crop
    with open(os.path.join(MVSO_PATH, 'image_url_mappings', 'english.csv'), 'r') as f:
        i = 0
        for line in f.readlines():
            if i == 0:      # skip header
                i += 1
                continue
            else:
                if i < 1785550:
                    i += 1
                    continue
                bc, url = line.strip().split(',')
                bc_dir = os.path.join(output_dir, bc)

                if i % 50 == 0:
                    time.sleep(0.1)
                    print 'bi_concept: {}; num_imgs: {}'.format(bc, i)
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

                urllib.urlretrieve(url, fp)

                # Reopen image to resize and central crop
                try:
                    im = Image.open(fp)
                    if im.mode != 'RGB':      # type L, P, etc. shows some type of Flickr unavailable photo img
                        os.remove(fp)
                        continue
                    im = np.array(im)
                    im =  mr.resize_and_center_crop(im, target_w, target_h)
                    Image.fromarray(im).save(fp)
                except Exception as e:
                    print e

if __name__ == '__main__':

    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='Download and process data')
    parser.add_argument('--bc_imgfps', dest='bc_imgfps', action='store_true')
    parser.add_argument('--bc_traintest', dest='bc_traintest', action='store_true')
    parser.add_argument('--move_bad_jpgs', dest='move_bad_jpgs', action='store_true')
    parser.add_argument('--you_dl_imgs', dest='you_dl_imgs', action='store_true')
    parser.add_argument('--mvso_sent', dest='mvso_sent', action='store_true')
    parser.add_argument('--mvso_emo', dest='mvso_emo', action='store_true')
    parser.add_argument('--mvso_dl_imgs', dest='mvso_dl_imgs', action='store_true')
    parser.add_argument('--sentibank_to_tfrecords', dest='sentibank_to_tfrecords', action='store_true')
    cmdline = parser.parse_args()

    if cmdline.bc_imgfps:
        bc2img_fps = get_all_bc_img_fps()
        print len([k for k, v in bc2img_fps.items() if len(v) > 250])
    elif cmdline.bc_traintest:
        pprint(get_bc_traintest())
    elif cmdline.move_bad_jpgs:
        move_bad_jpgs()
    elif cmdline.you_dl_imgs:
        retrieve_you_imemo_imgs()
    elif cmdline.mvso_sent:
        pprint(get_MVSO_bc2sentiment())
    elif cmdline.mvso_emo:
        pprint(get_MVSO_bc2emotion2value())
    elif cmdline.mvso_dl_imgs:
        download_MVSO_imgs()
    elif cmdline.sentibank_to_tfrecords:
        write_sentibank_to_tfrecords()
