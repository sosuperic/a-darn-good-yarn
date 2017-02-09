# Controller for viewing shape 
# TODO for refactor:
# 1) Consolidate data structures? clusters, ts, ts_idx2title, etc. all have 'films/shorts' as first key
# maybe just do data = {}, for format in formats: data['format] = {}
# And then attach the clusters, ts, etc.

from flask import Flask, Response, request, render_template
import json
import math
from natsort import natsorted
import numpy as np
import os
import pandas as pd
import pickle

from shape import app
from core.predictions.utils import smooth, DTWDistance
from core.utils.utils import get_credits_idx, AUDIO_SENT_PRED_FN, VIZ_SENT_PRED_FN

###### PATHS, ETC.
FORMATS = ['films', 'shorts', 'ads']

VIDEOS_PATH = 'shape/static/videos/'
OUTPUTS_DATA_PATH = 'shape/outputs/cluster/data/'

# For Clusters view:
TS_FN = \
    {'films': 'ts_dirfilms-n510-w1000-ds1-maxnf10000-fnsent_biclass_19.pkl',
     'shorts': 'ts_dirshorts-n1326-w0.14-ds1-maxnf1800-fnsent_biclass_19.pkl',
     'ads': None}
TS_IDX2TITLE_FN = \
    {'films': 'ts-idx2title_dirfilms-n510-w1000-ds1-maxnf10000-fnsent_biclass_19.pkl',
     'shorts': 'ts-idx2title_dirshorts-n1326-w0.14-ds1-maxnf1800-fnsent_biclass_19.pkl',
     'ads': None}
TS_MEAN_FN = \
    {'films': 'ts-mean_dirfilms-n510-w1000-ds1-maxnf10000-fnsent_biclass_19.pkl',
     'shorts': 'ts-mean_dirshorts-n1326-w0.14-ds1-maxnf1800-fnsent_biclass_19.pkl',
     'ads': None}
TS_STD_FN = \
    {'films': 'ts-std_dirfilms-n510-w1000-ds1-maxnf10000-fnsent_biclass_19.pkl',
     'shorts': 'ts-std_dirshorts-n1326-w0.14-ds1-maxnf1800-fnsent_biclass_19.pkl',
     'ads': None}
CENTROIDS_FN = \
    {'films': 'kmedoids-centroids_dirfilms-n510-w1000-ds1-maxnf10000-fnsent_biclass_19-k{}-it100-r250.pkl',
     'shorts': 'kmedoids-centroids_dirshorts-n1326-w0.14-ds1-maxnf1800-fnsent_biclass_19-k{}-it100-r45.pkl',
     'ads': None}
TS_DISTS_FN = \
    {'films': 'kmedoids-ts-dists_dirfilms-n510-w1000-ds1-maxnf10000-fnsent_biclass_19-k{}-it100-r250.pkl',
     'shorts': 'kmedoids-ts-dists_dirshorts-n1326-w0.14-ds1-maxnf1800-fnsent_biclass_19-k{}-it100-r45.pkl',
     'ads': None}

# TS_FN = \
#     {'films': 'ts_dirfilms-n510-w500-ds1-maxnf10000-fnsent_biclass_19.pkl',
#      'shorts': 'ts_dirshorts-n1326-wNone-ds1-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}
# TS_IDX2TITLE_FN = \
#     {'films': 'ts-idx2title_dirfilms-n510-w500-ds1-maxnf10000-fnsent_biclass_19.pkl',
#      'shorts': 'ts-idx2title_dirshorts-n1326-wNone-ds1-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}
# TS_MEAN_FN = \
#     {'films': 'ts-mean_dirfilms-n510-w500-ds1-maxnf10000-fnsent_biclass_19.pkl',
#      'shorts': 'ts-mean_dirshorts-n1326-wNone-ds1-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}
# TS_STD_FN = \
#     {'films': 'ts-std_dirfilms-n510-w500-ds1-maxnf10000-fnsent_biclass_19.pkl',
#      'shorts': 'ts-std_dirshorts-n1326-wNone-ds1-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}
# CENTROIDS_FN = \
#     {'films': 'centroids_dirfilms-n510-w500-ds1-maxnf10000-fnsent_biclass_19-k{}-it5-r250.pkl',
#      'shorts': 'centroids_dirshorts-n1326-wNone-ds1-maxnf1800-fnsent_biclass_19-k{}-it15-r45.pkl',
#      'ads': None}
# ASSIGNMENTS_FN = \
#     {'films': 'assignments_dirfilms-n510-w500-ds1-maxnf10000-fnsent_biclass_19-k{}-it5-r250.pkl',
#      'shorts': 'assignments_dirshorts-n1326-wNone-ds1-maxnf1800-fnsent_biclass_19-k{}-it15-r45.pkl',
#      'ads': None}
# TS_DISTS_FN = \
#     {'films': 'ts-dists_dirfilms-n510-w500-ds1-maxnf10000-fnsent_biclass_19-k{}-it5-r250.pkl',
#      'shorts': 'ts-dists_dirshorts-n1326-wNone-ds1-maxnf1800-fnsent_biclass_19-k{}-it5-r45.pkl',
#      'ads': None}


# OUTPUTS_DATA_PATH = 'shape/outputs/cluster/data/old1/'
# TS_FN = \
#     {'films': 'ts_dirfilms-n441-normMagsTrue-w1000-ds6-maxnfinf-fnsent_biclass_19.pkl',
#      'shorts': 'ts_dirshorts-n1323-normMagsTrue-w30-ds3-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}
# TS_IDX2TITLE_FN = \
#     {'films': 'ts_idx2title_dirfilms-n441-normMagsTrue-w1000-ds6-maxnfinf-fnsent_biclass_19.pkl',
#      'shorts': 'ts_idx2title_dirshorts-n1323-normMagsTrue-w30-ds3-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}
# CENTROIDS_FN = \
#     {'films': 'centroids_dirfilms-n441-normMagsTrue-k{}-w1000-ds6-maxnfinf-fnsent_biclass_19.pkl',
#      'shorts': 'centroids_dirshorts-n1323-normMagsTrue-k{}-w30-ds3-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}
# ASSIGNMENTS_FN = \
#     {'films': 'assignments_dirfilms-n441-normMagsTrue-k{}-w1000-ds6-maxnfinf-fnsent_biclass_19.pkl',
#      'shorts': 'assignments_dirshorts-n1323-normMagsTrue-k{}-w30-ds3-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}
# TS_DISTS_FN = \
#     {'films': 'ts_dists_dirfilms-n441-normMagsTrue-k{}-w1000-ds6-maxnfinf-fnsent_biclass_19.pkl',
#      'shorts': 'ts_dists_dirshorts-n1323-normMagsTrue-k{}-w30-ds3-maxnf1800-fnsent_biclass_19.pkl',
#      'ads': None}

###### GLOBALS
title2vidpath = {}
format2titles = {}
title2format = {}

# For One Video view - to adjust window length
title2pred_len = {}     # films/shorts -> title -> len
# For One Video view - used for initial curve to display
cur_format = None
cur_title = None
cur_viz_pd_df = None
cur_audio_pd_df = None
cur_vid_framepaths = None

# For Clusters view
clusters = {}           # films/shorts -> key (k) -> {assignments: k-idx: array, centroids: k-idx: array, closest: k-idx: array of member_indices}
ts = {}                 # films/shorts -> list of arrays
ts_idx2title = {}       # films/shorts -> idx -> title

########################################################################################################################
# PREDICTION FUNCTIONS
########################################################################################################################
def get_preds_from_dfs(name2df_and_window):
    """
    Return dict. Key is the type (visual/audio). Value is a dictionary that maps label to list of smoothed predictions.
    """
    preds = {}
    for name, df_and_window in name2df_and_window.items():
        df = df_and_window['df']
        window = df_and_window['window']
        preds[name] = {}
        if name == 'visual':
            values = list(smooth(df.pos.values, window_len=window)) if (df is not None) else []
            preds[name]['visual-valence'] = values
        elif name == 'audio':
            values = list(smooth(df.Valence.values, window_len=window)) if (df is not None) else []
            preds[name]['audio-valence'] = values
    return preds

# def get_preds_from_df(df, window_len=48):
#     """
#     Return dict. Each key is the label, value is list of smoothed frame-wise predictions
#     """
#     if len(df.columns) == 2:     # sent_biclass -> just show positive
#         preds = {'pos': list(smooth(df.pos.values, window_len=window_len))}
#     else:
#         preds = {}
#         for c in df.columns:
#             preds[c] = list(smooth(df[c].values, window_len=window_len))
#     return preds

def get_cur_vid_df_and_framepaths(cur_title):
    """
    Return df and list of framepaths. Used for ajax call in One Video view that retrieves smoothed predictions for
    new video. Framepaths are now relative to VIDEOS_PATH (which is a static path for js) instead of the full
    path so that the HTML template can display it.
    """
    global title2vidpath, cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths

    # Get dataframe
    viz_preds_path = os.path.join(title2vidpath[cur_title], 'preds', VIZ_SENT_PRED_FN)
    if os.path.exists(viz_preds_path):
        cur_viz_pd_df = pd.read_csv(viz_preds_path)
    else:
        cur_viz_pd_df = None

    audio_preds_path = os.path.join(title2vidpath[cur_title], 'preds', AUDIO_SENT_PRED_FN)
    if os.path.exists(audio_preds_path):
        print audio_preds_path
        cur_audio_pd_df = pd.read_csv(audio_preds_path)
    else:
        cur_audio_pd_df = None

    # Get framepaths
    cur_vid_framepaths = [f for f in os.listdir(os.path.join(title2vidpath[cur_title], 'frames')) if \
            not f.startswith('.')]
    cur_vid_framepaths = [os.path.join(title2vidpath[cur_title].split(VIDEOS_PATH)[1], 'frames', f) for \
        f in cur_vid_framepaths]
    cur_vid_framepaths = natsorted(cur_vid_framepaths)

    # Ignore credits
    vidpath = title2vidpath[cur_title]
    credit_idx = get_credits_idx(vidpath)
    if credit_idx:
        if cur_viz_pd_df is not None:
            cur_viz_pd_df = cur_viz_pd_df[:credit_idx]
        if cur_audio_pd_df is not None:
            cur_audio_pd_df = cur_audio_pd_df[:credit_idx]
        cur_vid_framepaths = cur_vid_framepaths[:credit_idx]

    return cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths

def get_cur_mp3path(cur_vid_framepaths):
    # Get current mp3path if it exists
    cur_mp3path = ''
    vid_dirpath = os.path.dirname(os.path.dirname(cur_vid_framepaths[0]))
    for f in os.listdir(os.path.join(VIDEOS_PATH, vid_dirpath)):
        if f.endswith('mp3'):
            cur_mp3path = os.path.join(vid_dirpath, f)
    return cur_mp3path

########################################################################################################################
# INITIAL SETUP
########################################################################################################################
def get_all_vidpaths_with_frames_and_preds():
    """
    Return list of full paths to every video directory that a) contains frames/ directory, b) predictions/ directory,
    and c) frames/ directory has more than 0 frames. Starts walking in VIDEOS_PATH directory. Each full path is of the
    form '<VIDEOS_PATH>/@Animated/@OldDisney/Feast/'.
    """
    def root_contains_valid_fmt(root):
        for fmt in FORMATS:
            if fmt in root:
                return True
        return False

    def walk_dirpath(dirpath):
        vidpaths = []
        for root, dirs, files in os.walk(dirpath):
            if not root_contains_valid_fmt(root):
                continue

            if ('frames' in os.listdir(root)) and ('preds' in os.listdir(root)) \
                and (len(os.listdir(os.path.join(root, 'frames'))) > 0):
                vidpaths.append(root)
        return vidpaths

    vidpaths = []
    for fmt in FORMATS:
        # print fmt, vidpaths
        vidpaths.extend(walk_dirpath(os.path.join(VIDEOS_PATH, fmt)))

    return vidpaths

def setup_initial_data():
    """
    Return initial data to pass to template.
    Also set global variables cur_viz_pd_df and cur_title
    """
    ####################################################################################################################
    # One video view (primarily) - set titles, videopaths, etc
    ####################################################################################################################
    global title2vidpath, format2titles, title2format

    vidpaths = get_all_vidpaths_with_frames_and_preds()
    for fmt in FORMATS:
        format2titles[fmt] = []
    for vp in vidpaths:
        format = vp.split(VIDEOS_PATH)[1].split('/')[0]     # shape/static/videos/shorts/shortoftheweek/Feast -> shorts
        t = os.path.basename(vp)
        title2format[t] = format
        format2titles[format].append(t)
        title2vidpath[t] = vp
    for format, titles in format2titles.items():
        format2titles[format] = sorted(titles)
        print '{}: {}'.format(format, len(titles))

    ####################################################################################################################
    # Cluster view - set time series, cluster related data
    ####################################################################################################################
    # All time series
    global ts, ts_idx2title, title2pred_len
    # NOTE: all the time series are of the same length -- this is the saved interpolated time series used during
    # clustering. These are used to display the closest movies in the clusters view

    fmt2mean, fmt2std = {}, {}
    for fmt in ['films', 'shorts']:#FORMATS:
        try:
            # Load mean and std to unnormalize time series
            mean_fn = TS_MEAN_FN[fmt]
            std_fn = TS_STD_FN[fmt]
            mean_path = os.path.join(OUTPUTS_DATA_PATH, mean_fn)
            std_path = os.path.join(OUTPUTS_DATA_PATH, std_fn)
            with open(mean_path) as f:
                mean = pickle.load(f)
                mean = mean.mean()
                fmt2mean[fmt] = mean
            with open(std_path) as f:
                std = pickle.load(f)
                std = std.mean()
                fmt2std[fmt] = std

            # Clusters view
            ts[fmt] = pickle.load(open(os.path.join(OUTPUTS_DATA_PATH, TS_FN[fmt]), 'rb'))
            ts[fmt] = [ts[fmt][i] * fmt2std[fmt] + fmt2mean[fmt] for i in range(len(ts[fmt]))]   # unnormalize
            ts[fmt] = [list(arr) for arr in ts[fmt]]        # make it serializable
            ts_idx2title[fmt] = pickle.load(open(os.path.join(OUTPUTS_DATA_PATH, TS_IDX2TITLE_FN[fmt]), 'rb'))

            # One video view - adjusting window size
            title2pred_len[fmt] = {}

            for ts_idx, title in ts_idx2title[fmt].items():
                if title in title2vidpath:
                    vid_framespath = os.path.join(title2vidpath[title], 'frames')
                    title2pred_len[fmt][title] = len(os.listdir(vid_framespath))
        except Exception as e:
            print fmt, e

    global clusters
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for fmt in FORMATS:
        clusters[fmt] = {}
        for k in ks:
            try:
                k = str(k)          # use string so it's treated as a js Object instead of an array in template
                centroids_fn = CENTROIDS_FN[fmt].format(k)
                ts_dists_fn = TS_DISTS_FN[fmt].format(k)
                centroids_path = os.path.join(OUTPUTS_DATA_PATH, centroids_fn)
                ts_dists_path = os.path.join(OUTPUTS_DATA_PATH, ts_dists_fn)
                if os.path.exists(centroids_path) and os.path.exists(ts_dists_path):
                    clusters[fmt][k] = {}
                    with open(centroids_path, 'rb') as f:
                        clusters[fmt][k]['centroids'] = pickle.load(f)

                    # TS Distances to centroids
                    # ts_dists: dict, key is int (centroid_idx), value = dict (key is member_idx, value is distance)
                    ts_dists_fn = TS_DISTS_FN[fmt].format(k)
                    ts_dists_path = os.path.join(OUTPUTS_DATA_PATH, ts_dists_fn)
                    if os.path.exists(ts_dists_path):
                        with open(ts_dists_path) as f:
                            kdists = pickle.load(f)      # key is ts_index, value is distance to its centroid
                        centroid2closest = {}
                        for centroid_idx, dists in kdists.items():
                            sorted_member_indices = sorted(dists, key=dists.get)
                            top_n = sorted_member_indices[:10]
                            # top_n_series = [ts['films'][m_idx] for m_idx in top_n]
                            centroid2closest[centroid_idx] = top_n
                        clusters[fmt][k]['closest'] = centroid2closest

                else:
                    print 'Centroids/assignments/ts_dists path doesnt exist:\n{}\n{}'.format(
                        centroids_path, ts_dists_path)
            except Exception as e:
                print e

    # tmp()

    print 'Setup done'

# def tmp():
#     # DEBUGING
#     global format2titles, title2pred_len, \
#         cur_format, cur_title, cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths, \
#         clusters, ts_idx2title, ts
#
#     # Get information for *first* video to show
#     # cur_format = format2titles.keys()[0]
#     cur_format = 'films'
#     cur_title = format2titles[cur_format][0]
#     cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths = get_cur_vid_df_and_framepaths(cur_title)
#     cur_vid_preds = get_preds_from_dfs({'visual': {'df': cur_viz_pd_df, 'window': 300},  # window_len has to match default in html file
#                                         'audio': {'df': cur_audio_pd_df, 'window': 1}})
#
#     import logging
#     logging.debug(cur_format)
#     print cur_format
#     print cur_title
#     print cur_viz_pd_df
#     print cur_audio_pd_df
#     print cur_vid_preds
#     data = {'format2titles': format2titles,
#             'framepaths': cur_vid_framepaths,
#             'preds': cur_vid_preds,
#             'title2pred_len': title2pred_len,
#             # Clusters view
#             'clusters': clusters,
#             'ts_idx2title': ts_idx2title,
#             'ts': ts}


#################################################################################################
# ROUTING FUNCTIONS
#################################################################################################
@app.route('/shape', methods=['GET'])
def shape():
    """
    Return main shape template with data

    Data
    ----
    format2titles:
        - to create dropdowns in dat.gui
    framepaths:
        - to display image for current video
    preds:
        - to create graph for current video
    clusters:
        - for clusters view
    """
    global format2titles, title2pred_len, \
        cur_format, cur_title, cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths, \
        clusters, ts_idx2title, ts

    # Get information for *first* video to show
    # cur_format = format2titles.keys()[0]
    cur_format = 'films'
    cur_title = format2titles[cur_format][0]
    cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths = get_cur_vid_df_and_framepaths(cur_title)
    cur_vid_preds = get_preds_from_dfs({'visual': {'df': cur_viz_pd_df, 'window': 600},  # window_len has to match default in html file
                                        'audio': {'df': cur_audio_pd_df, 'window': 300}})
    cur_mp3path = get_cur_mp3path(cur_vid_framepaths)

    data = {'format2titles': format2titles,
            'framepaths': cur_vid_framepaths,
            'preds': cur_vid_preds,
            'title2pred_len': title2pred_len,
            'mp3path': cur_mp3path,
            # Clusters view
            'clusters': clusters,
            'ts_idx2title': ts_idx2title,
            'ts': ts}

    return render_template('plot_shape.html', data=json.dumps(data))

@app.route('/api/pred/<title>/<visual_window>/<audio_window>', methods=['GET'])
def get_preds_and_frames(title, visual_window, audio_window):
    """
    Return predictions for a given movie with window_len; update global vars
    """
    global title2vidpath, \
        cur_format, cur_title, cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths
    if title != cur_title:
        title = title.encode('utf-8')
        cur_format = title2format[title]
        cur_title = title
        cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths = get_cur_vid_df_and_framepaths(cur_title)

    # print window_len, len(cur_viz_pd_df)
    # preds = get_preds_from_df(cur_viz_pd_df, window_len=int(window_len))
    cur_vid_preds = get_preds_from_dfs({'visual': {'df': cur_viz_pd_df, 'window': int(visual_window)},  # window_len has to match default in html file
                                        'audio': {'df': cur_audio_pd_df, 'window': int(audio_window)}})
    cur_mp3path = get_cur_mp3path(cur_vid_framepaths)

    data = {'preds': cur_vid_preds, 'framepaths': cur_vid_framepaths, 'mp3path': cur_mp3path}

    return Response(
        json.dumps(data),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )

setup_initial_data()
