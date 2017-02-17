# Prepare the data (parse, download, etc.)

import argparse
import cPickle as pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from pprint import pprint
import re
from senti_classifier import senti_classifier
import sqlite3
import sys

from core.utils.utils import  CMU_PATH, VIDEOPATH_DB, VIDEOMETADATA_DB

# encoding = utf8
reload(sys)
sys.setdefaultencoding('utf8')

########################################################################################################################
# Utilities, functions, etc.
########################################################################################################################
def split_into_sentences(text):
    """
    Split body of text into sentences. Returns list of strs.
    """

    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]

    return sentences

def classify_sentences(sents):
    """
    Return list of tuples (sentence, positive, negative)
    """
    results = []
    for sent in sents:
        pos_score, neg_score = senti_classifier.polarity_scores([sent])
        results.append([sent, pos_score, neg_score])

    return results

########################################################################################################################
# Sentence-level sentiment analysis on CMU plot summaries
########################################################################################################################
def text_sent_on_film_summaries(overwrite):
    """
    Save sentence-level sentiments for each film
    """
    # Get videos, CMU movie ids, paths, summaries
    print 'Loading VideoMetadata DB'
    vmd = pickle.load(open(VIDEOMETADATA_DB))

    print 'Looking up paths for each video from VideoPath DB'
    video2path = {}
    conn = sqlite3.connect(VIDEOPATH_DB)
    with conn:
        cur = conn.cursor()
        rows = cur.execute("SELECT title, dirpath FROM VideoPath WHERE category=='films'")
        for row in rows:
            video2path[row[0]] = row[1]  # title includes year, e.g. Serenity (2003)

    print 'Getting summaries for all CMU movies'
    id2summary = {}
    for line in open(os.path.join(CMU_PATH, 'plot_summaries.txt'), 'rb').readlines():
        split = line.split('\t')
        id2summary[split[0]] = split[1]

    # Get summary for each video and compute sentiment analysis
    print 'Computing sentence-level sentiment for each film in corpus'
    num_videos = 0
    for video, md in vmd['films'].items():
        try:
            # Get some paths
            path = video2path[video]
            preds_path = os.path.join(path, 'preds')
            preds_out_fp = os.path.join(preds_path, 'text_summary.csv')
            fig_out_fp = os.path.join(preds_path, 'text_summary.png')

            # Skip if not overwriting and already exists
            if (not overwrite) and (os.path.exists(preds_out_fp)) and (os.path.exists(fig_out_fp)):
                continue

            # Calculate sentiment
            print '{}: Saving to preds_path: {}'.format(num_videos, preds_path)
            id = md['id']
            summary = id2summary[id]
            sentences = split_into_sentences(summary)
            sentiments = classify_sentences(sentences)

            # Save
            if not os.path.exists(preds_path):
                os.mkdir(preds_path)

            # Save csv with sentences and preds
            with open(preds_out_fp, 'wb') as f:
                f.write('idx,sentence,pos,neg\n')
                i = 0
                for sentence, pos, neg in sentiments:
                    f.write('{},{},{},{}\n'.format(i, sentence, pos, neg))
                    i += 1
                # (Write idx so it's that much easier to compare against figure)

            # Save figure
            plot_x = range(len(sentences))
            positives = [s[1] for s in sentiments]
            negatives = [s[2] for s in sentiments]
            plt.plot(plot_x, positives, label='pos')
            plt.plot(plot_x, negatives, label='neg')
            plt.legend()
            plt.savefig(fig_out_fp)

            num_videos += 1

        except Exception as e:      # probably movie missing in one of dictionaries
            print e
            continue

    print 'Computed sentiments for {} videos'.format(num_videos)


if __name__ == '__main__':

    # Set up commmand line arguments
    parser = argparse.ArgumentParser(description='Download and process data')
    parser.add_argument('--text_sent_on_film_summaries', dest='text_sent_on_film_summaries', action='store_true')
    parser.add_argument('--summaries_overwrite', dest='summaries_overwrite', action='store_true',
                        help='skip video if this is false')

    cmdline = parser.parse_args()

    if cmdline.text_sent_on_film_summaries:
        text_sent_on_film_summaries(cmdline.summaries_overwrite)
