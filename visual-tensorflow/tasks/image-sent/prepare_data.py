# Prepare the data

from collections import defaultdict, Counter
import numpy as np
import os
import pickle
import re
from skimage import color
import urllib

from PIL import Image

from core.utils.MovieReader import MovieReader

# Sentibank - bi_concepts1553: mapping ajdective noun pairs to sentiment
BC_PATH = 'data/Sentibank/Flickr/bi_concepts1553'
BC_TRAINTEST_PATH = 'data/Sentibank/Flickr/trainingAndTestingPartition'
EMOLEX_PATH = 'data/emolex/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'

# You dataset - 20k images with emotions
YOU_IMEMO_PATH = 'data/you_imemo/agg'

# MVSO dataset - mutlilingual, larger version of Sentibank VSO; also has emotions
MVSO_PATH = 'data/MVSO'


### bi_concepts1553 related data munging

def get_all_bc_img_fps():
    """Return dictionary mapping bi_concept to list of img file paths"""
    bc2img_fps = {}
    for bc in [d for d in os.listdir(BC_PATH) if not d.startswith('.')]:
        cur_bc_path = os.path.join(BC_PATH, bc)
        img_fns = [f for f in os.listdir(cur_bc_path) if f.endswith('jpg')]
        img_fps = [os.path.join(cur_bc_path, fn) for fn in img_fns]
        bc2img_fps[bc] = img_fps

    return bc2img_fps

def get_bc_sentiments_and_counts():
    """Return dictionary mapping bi_concept to positive-negative polarity values"""
    bc2sent_and_count = defaultdict(dict)
    with open('data/Sentibank/VSO/3244ANPs.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.endswith(']') and '_' in line:
                print line
                m = re.match(r'(.+_.+)\s\[sentiment:\s(.+)\]\s\[#imgs:\s(.+)\]', line)
                bc, sent, count = m.group(1), float(m.group(2)), int(m.group(3).replace(',', ''))
                bc2sent_and_count[bc]['sent'] = sent
                bc2sent_and_count[bc]['count'] = count
    return bc2sent_and_count

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

def get_bc2emotions():
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

    bc2emotions = defaultdict(list)
    bc2img_fps = get_all_bc_img_fps()
    word2emotions = get_emolex()
    for bc, _ in bc2img_fps.items():
        # print bc
        adj, noun = bc.split('_')
        if adj in word2emotions:
            for emotion in word2emotions[adj]:
                bc2emotions[bc].append(emotion)
        if noun in word2emotions:
            for emotion in word2emotions[noun]:
                bc2emotions[bc].append(emotion)
        bc2emotions[bc] = Counter(bc2emotions[bc])

    return bc2emotions


# You image emotion dataset
def get_you_imemo_urls():
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
    emo2urls = get_you_imemo_urls()
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


### MVSO
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
    mr = MovieReader()          # used to resize and center crop
    with open(os.path.join(MVSO_PATH, 'image_url_mappings', 'english.csv'), 'r') as f:
        i = 0
        for line in f.readlines():
            if i == 0:      # skip header
                i += 1
                continue
            else:
                bc, url = line.strip().split(',')
                bc_dir = os.path.join(output_dir, bc)

                if i % 50 == 0:
                    print 'bi_concept: {}; num_imgs: {}'.format(bc, i)

                # Make bi_concept directory if it doesn't exist
                if not os.path.exists(bc_dir):
                    os.makedirs(bc_dir)

                # Retrive image and save
                fn = os.path.basename(url)
                fp = os.path.join(bc_dir, fn)
                urllib.urlretrieve(url, fp)

                # Reopen image to resize and central crop
                im = Image.open(fp)
                if im.mode != 'RGB':      # type L, P, etc. shows some type of Flickr unavailable photo img
                    os.remove(fp)
                    continue
                im = np.array(im)
                im =  mr.resize_and_center_crop(im, target_w, target_h)
                Image.fromarray(im).save(fp)

                i += 1

### BC
# get_all_bc_img_fps()
# tmp = get_bc_sentiments_and_counts()
# get_bc_traintest()
# get_bc2emotions()

### You im_emo
# get_you_imemo_urls()
# retrieve_you_imemo_imgs()

### MVSO
import pprint
# bc2sentiment = get_MVSO_bc2sentiment()
# pprint.pprint(bc2sentiment)
# print len(bc2sentiment.keys())
# bc2emo2val = get_MVSO_bc2emotion2value()
# pprint.pprint(bc2emo2val)
download_MVSO_imgs()