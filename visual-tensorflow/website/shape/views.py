# Controller for viewing shape 

import json
import os
import time
from flask import Flask, Response, request, render_template
from shape import app

import uuid
import csv
import pickle
import random
import string
import re
import os
import numpy as np
import pandas as pd


from natsort import natsorted

VIDEOS_PATH = 'data/videos/'

title2vidpath = {}
cur_pd_df = None
cur_title = None

#################################################################################################
# NON-ROUTING FUNCTIONS 
#################################################################################################
def smooth(x, window_len=48, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2 * x[0] - x[window_len-1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:  
        w=eval('np.' + window + '(window_len)')
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len:-window_len+1]


def get_preds_from_df(df, window_len=48):
    """Return smoothed preds: each key is the label, value is list of frame-wise predictions"""
    if len(df.columns) == 2:     # sent_biclass -> just show positive
        preds = {'pos': list(smooth(df.pos.values, window_len=window_len))}
    else:
        preds = {}
        for c in df.columns:
            preds[c] = list(smooth(df[c].values, window_len=window_len))
    return preds

#################################################################################################
# INITIAL SETUP
#################################################################################################
def get_all_vidpaths_with_preds():
    """
    Return list of full paths to every video directory that contains predictions
    e.g. [<VIDEOS_PATH>/@Animated/@OldDisney/Feast/, ...]
    """
    vidpaths = []
    for root, dirs, files in os.walk(VIDEOS_PATH):
        if ('frames' in os.listdir(root)) and ('preds' in os.listdir(root)):
            vidpaths.append(root)
    return vidpaths

def get_initial_data():
    """
    Return initial data to pass to template. 
    Also set global variables cur_pd_df and cur_title
    """
    vidpaths = get_all_vidpaths_with_preds()
    titles = []
    global title2vidpath
    for vp in vidpaths:
        t = os.path.basename(vp)
        titles.append(t)
        title2vidpath[t] = vp
    print title2vidpath

    # Get paths every frame for every video
    title2framepaths = {}
    for i, t in enumerate(titles):
        vp = vidpaths[i]
        title2framepaths[t] = natsorted([os.path.join(vp, 'frames', f) for f in os.listdir(os.path.join(vp, 'frames')) \
        if not f.startswith('.')])

    # Get initial preds (first video to show)
    global cur_pd_df
    cur_pd_df = pd.read_csv(os.path.join(vidpaths[0], 'preds', 'sent_biclass.csv'))
    preds = get_preds_from_df(cur_pd_df, window_len=48)
    
    # Now turn full paths to relative paths because js has relative path starting from videos/
    vidpaths = [vp.lstrip(VIDEOS_PATH) for vp in vidpaths]
    title2framepaths = {t: [fp.lstrip(VIDEOS_PATH) for fp in vid] for t, vid in title2framepaths.items()}

    global cur_title
    cur_title = titles[0]

    data = {'titles': titles, 'title2framepaths': title2framepaths, 'preds': preds}
    return data

#################################################################################################
# ROUTING FUNCTIONS 
#################################################################################################
@app.route('/shape', methods=['GET'])
def shape():
    """Main shape template with initial data"""
    data = get_initial_data()
    return render_template('plot_shape.html', data=json.dumps(data))

@app.route('/api/pred/<title>/<window_len>', methods=['GET'])
def get_preds(title, window_len):
    """Return predictions for a given movie with window_len, update global vars"""
    global cur_pd_df, cur_title, title2vidpath
    if title != cur_title:
        cur_title = title
        vidpath = title2vidpath[title]
        cur_pd_df = pd.read_csv(os.path.join(vidpath, 'preds', 'sent_biclass.csv'))

    preds = get_preds_from_df(cur_pd_df, window_len=int(window_len))

    return Response(
        json.dumps(preds),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )
