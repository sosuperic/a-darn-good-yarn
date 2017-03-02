# Controller for viewing shape 
# TODO for refactor:
# 1) Consolidate data structures? clusters, ts, ts_idx2title, etc. all have 'films/shorts' as first key
# maybe just do data = {}, for format in formats: data['format] = {}
# And then attach the clusters, ts, etc.

from flask import Flask, Response, request, render_template
import json
from natsort import natsorted
import os
import pandas as pd
import pickle

from shape import app
from core.predictions.utils import smooth
from core.utils.utils import get_credits_idx, AUDIO_SENT_PRED_FN, VIZ_SENT_PRED_FN

### PARAMS ###
FORMATS = ['films', 'shorts', 'ads']
# TODO: want to pass LOAD_CLUSTERS as a command line argument, but need extra wrangling to work with gunicorn
LOAD_CLUSTERS = False
CLUSTERS_KS = [2, 3, 4  , 5, 6, 7, 8, 9, 10]

### PATHS ###
VIDEOS_PATH = 'shape/static/videos/'
OUTPUTS_DATA_PATH = 'shape/outputs/cluster/data/'

# Clusters view paths
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

### GLOBALS ###
# One Video view
# For all videos
title2vidpath = {}
format2titles = {fmt: [] for fmt in FORMATS}
title2format = {}
title2pred_len = {fmt: {} for fmt in FORMATS}     # fmt -> title -> len
# For current video
cur_title = None
cur_viz_pd_df = None
cur_audio_pd_df = None
cur_vid_framepaths = None
cur_mp3path = None
# To adjust window length when switching between videos

# Clusters view
clusters = {}           # fmt -> key (k) -> value {assignments: k-idx: array, centroids: k-idx: array, closest: k-idx: array of member_indices}
ts = {}                 # fmt -> list of arrays
ts_idx2title = {}       # fmt -> idx -> title

########################################################################################################################
# One Video view - get predictions, frames, audio, etc.
########################################################################################################################
def get_preds_from_dfs(name2df_and_window):
    """
    Return dict. Key is the type (visual/audio). Value is a dictionary that maps label to list of smoothed predictions.
    """
    def get_capped_std_val(mean_val, std_val):
        ceiling_capped = floor_capped = std_val
        if (std_val + mean_val > 1):
            ceiling_capped = 1 - mean_val
        if (mean_val - std_val < 0):
            floor_capped = mean_val
        capped_std = min(ceiling_capped, floor_capped)
        return capped_std

    preds = {}
    for name, df_and_window in name2df_and_window.items():
        df = df_and_window['df']
        window = df_and_window['window']
        preds[name] = {}
        if name == 'visual':
            values = list(smooth(df.pos.values, window_len=window)) if (df is not None) else []
            preds[name]['pos'] = values
            preds[name]['std'] = [0.04 for _ in range(len(values))]
        elif name == 'audio':
            values = list(smooth(df.Valence_mean.values, window_len=window)) if (df is not None) else []
            std = list(smooth(df.Valence_std.values, window_len=window)) if (df is not None) else []
            std = [get_capped_std_val(values[i], std_val) for i, std_val in enumerate(std)]
            preds[name]['pos'] = values
            preds[name]['std'] = std
            preds[name]['pos_lower'] = [max(0, values[i] - std_val) for i, std_val in enumerate(std)]
            preds[name]['pos_upper'] = [min(1, values[i] + std_val) for i, std_val in enumerate(std)]
    return preds

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
        cur_audio_pd_df = pd.read_csv(audio_preds_path)
    else:
        cur_audio_pd_df = None

    # Get framepaths
    # Note: vps in title2vidpath is of the form '<VIDEOS_PATH>/films/animated/Frozen (2013)/...'
    # Return the path relative to <VIDEOS_PATH>, i.e. 'films/animated/Frozen (2013)/...' and let
    # the template create the path relative to its location
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

def get_cur_mp3path(title):
    """
    Get mp3path of current video if it exists.

    Note: vps in title2vidpath is of the form '<VIDEOS_PATH>/films/animated/Frozen (2013)/...'
    Return the path relative to <VIDEOS_PATH>, i.e. 'films/animated/Frozen (2013)/...' and let
    the template create the path relative to its location
    """
    global title2vidpath
    cur_mp3path = ''
    vp = title2vidpath[title]
    for f in os.listdir(vp):
        if f.endswith('mp3'):
            cur_mp3path = os.path.join(vp, f)
            cur_mp3path = cur_mp3path.split(VIDEOS_PATH)[1]
    return cur_mp3path

########################################################################################################################
# Initial setup
########################################################################################################################
def get_all_valid_vidpaths():
    """
    Return list of full paths to every video directory that a) contains frames/ directory, b) predictions/ directory,
    and c) frames/ directory has more than 0 frames. Starts walking in VIDEOS_PATH directory. Each full path is of the
    form '<VIDEOS_PATH>/films/animated/Frozen (2013)/...'
    """
    def root_contains_valid_fmt(root):
        for fmt in FORMATS:
            if fmt in root:
                return True
        return False

    def walk_dirpath(dirpath):
        vidpaths_nframes = []
        for root, dirs, files in os.walk(dirpath):
            if not root_contains_valid_fmt(root):
                continue

            if ('frames' in os.listdir(root)) and ('preds' in os.listdir(root)):
                if (AUDIO_SENT_PRED_FN in os.listdir(os.path.join(root, 'preds'))) and \
                    (VIZ_SENT_PRED_FN in os.listdir(os.path.join(root, 'preds'))):
                    nframes = len(os.listdir(os.path.join(root, 'frames')))
                    if nframes > 0:
                        vidpaths_nframes.append([root, nframes])
        return vidpaths_nframes

    vidpaths_nframes = []
    for fmt in FORMATS:
        vidpaths_nframes.extend(walk_dirpath(os.path.join(VIDEOS_PATH, fmt)))

    return vidpaths_nframes

def setup_initial_data(load_clusters=False):
    """
    Setup globals
    """
    print 'Setting up initial data'

    ####################################################################################################################
    # One Video view (primarily) - set titles, videopaths, etc
    ####################################################################################################################
    global title2vidpath, format2titles, title2format, title2pred_len

    print 'Loading One Video view data'

    vidpaths_nframes = get_all_valid_vidpaths()
    for vp, nframes in vidpaths_nframes:
        fmt = vp.split(VIDEOS_PATH)[1].split('/')[0]     # shape/static/videos/shorts/shortoftheweek/Feast -> shorts
        t = os.path.basename(vp)

        title2vidpath[t] = vp
        format2titles[fmt].append(t)
        title2format[t] = fmt
        title2pred_len[t] = nframes

    # Sort titles
    for fmt, titles in format2titles.items():
        format2titles[fmt] = sorted(titles)
        print '{}: {}'.format(fmt, len(titles))


    ####################################################################################################################
    # Cluster view - set time series, cluster related data
    ####################################################################################################################
    if load_clusters:
        print 'Loading Clusters view data'

        global ts, ts_idx2title
        # All time series
        # NOTE: all the time series are of the same length -- this is the saved interpolated time series used during
        # clustering. These are used to display the closest movies in the clusters view

        fmt2mean, fmt2std = {}, {}
        for fmt in FORMATS:
            try:
                # Load mean and std to unnormalize time series
                mean_fn = TS_MEAN_FN[fmt]
                std_fn = TS_STD_FN[fmt]
                mean_path = os.path.join(OUTPUTS_DATA_PATH, mean_fn)
                std_path = os.path.join(OUTPUTS_DATA_PATH, std_fn)
                with open(mean_path) as f:
                    mean = pickle.load(f).mean()
                    fmt2mean[fmt] = mean
                with open(std_path) as f:
                    std = pickle.load(f).mean()
                    fmt2std[fmt] = std

                # Load time series and unnormalize each one
                ts[fmt] = pickle.load(open(os.path.join(OUTPUTS_DATA_PATH, TS_FN[fmt]), 'rb'))
                ts[fmt] = [ts[fmt][i] * fmt2std[fmt] + fmt2mean[fmt] for i in range(len(ts[fmt]))]
                ts[fmt] = [list(arr) for arr in ts[fmt]]        # make it serializable
                ts_idx2title[fmt] = pickle.load(open(os.path.join(OUTPUTS_DATA_PATH, TS_IDX2TITLE_FN[fmt]), 'rb'))

            except Exception as e:
                # print fmt, e
                pass

        global clusters
        for fmt in FORMATS:
            clusters[fmt] = {}
            for k in CLUSTERS_KS:
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
                except Exception as e:
                    # print e
                    pass

    print 'Setup done'

#################################################################################################
# Routing functions
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
        cur_title, cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths, cur_mp3path, \
        clusters, ts_idx2title, ts

    # One video view - get information for first video to show
    cur_title = format2titles[FORMATS[0]][0]
    cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths = get_cur_vid_df_and_framepaths(cur_title)
    cur_mp3path = get_cur_mp3path(cur_title)
    cur_vid_preds = get_preds_from_dfs(
        {'visual': {'df': cur_viz_pd_df, 'window': 600},  # window_len has to match default in html file
        'audio': {'df': cur_audio_pd_df, 'window': 600}})

    data = {
            # One Video view
            # All videos
            'format2titles': format2titles,
            'title2pred_len': title2pred_len,
            # Current video
            'framepaths': cur_vid_framepaths,
            'mp3path': cur_mp3path,
            'preds': cur_vid_preds,
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
        cur_title, cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths, cur_mp3path

    # Update current if it's a new title
    if title != cur_title:
        cur_title = title.encode('utf-8')
        cur_viz_pd_df, cur_audio_pd_df, cur_vid_framepaths = get_cur_vid_df_and_framepaths(cur_title)
        cur_mp3path = get_cur_mp3path(cur_title)

    cur_vid_preds = get_preds_from_dfs(
        {'visual': {'df': cur_viz_pd_df, 'window': int(visual_window)},
        'audio': {'df': cur_audio_pd_df, 'window': int(audio_window)}})


    data = {'preds': cur_vid_preds, 'framepaths': cur_vid_framepaths, 'mp3path': cur_mp3path}

    return Response(
        json.dumps(data),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )

# TODO: want to call this command line argument, but needs extra wrangling to work with gunicorn
# (See run.py and commit for some more context)
setup_initial_data(load_clusters=LOAD_CLUSTERS)