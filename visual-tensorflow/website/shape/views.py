# Controller for viewing shape 


import json
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from flask import Flask, Response, request, render_template

from shape import app
from core.predictions.utils import smooth

VIDEOS_PATH = 'shape/static/videos/'

title2vidpath = {}
titles = []
cur_pd_df = None
cur_title = None
cur_vid_framepaths = None

#################################################################################################
# PREDICTION FUNCTIONS 
#################################################################################################
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

def get_cur_vid_framepaths():
    global cur_vid_framepaths
    cur_vid_framepaths = [f for f in os.listdir(os.path.join(title2vidpath[cur_title], 'frames')) if \
            not f.startswith('.')]
    cur_vid_framepaths = [os.path.join(title2vidpath[cur_title].lstrip(VIDEOS_PATH), 'frames', f) for \
        f in cur_vid_framepaths]
    cur_vid_framepaths = natsorted(cur_vid_framepaths)
    return cur_vid_framepaths

def setup_initial_data():
    """
    Return initial data to pass to template. 
    Also set global variables cur_pd_df and cur_title
    """
    global title2vidpath, titles, cur_title, cur_pd_df, cur_vid_framepaths

    vidpaths = get_all_vidpaths_with_preds()
    for vp in vidpaths:
        t = os.path.basename(vp)
        titles.append(t)
        title2vidpath[t] = vp
    titles = sorted(titles)
    print titles
    cur_title = titles[0]

    # Get initial preds (first video to show)
    cur_pd_df = pd.read_csv(os.path.join(vidpaths[0], 'preds', 'sent_biclass.csv'))
    cur_vid_preds = get_preds_from_df(cur_pd_df, window_len=48)
    
    # Now turn full paths to relative paths because js has relative path starting from videos/
    # vidpaths = [vp.lstrip(VIDEOS_PATH) for vp in vidpaths]
    # title2framepaths = {t: [fp.lstrip(VIDEOS_PATH) for fp in vid] for t, vid in title2framepaths.items()}
    cur_vid_framepaths = get_cur_vid_framepaths()

    print 'done'

#################################################################################################
# ROUTING FUNCTIONS 
#################################################################################################
@app.route('/shape', methods=['GET'])
def shape():
    """Main shape template with initial data"""
    global titles, cur_pd_df, cur_vid_framepaths
    cur_vid_preds = get_preds_from_df(cur_pd_df, window_len=48)
    # titles to create dropdown in dat.gui
    # framepaths to display image for current video
    # preds to create graph for current video
    data = {'titles': titles, 'framepaths': cur_vid_framepaths, 'preds': cur_vid_preds}
    return render_template('plot_shape.html', data=json.dumps(data))

@app.route('/api/pred/<title>/<window_len>', methods=['GET'])
def get_preds_and_frames(title, window_len):
    """Return predictions for a given movie with window_len, update global vars"""
    global title2vidpath, cur_pd_df, cur_title, cur_vid_framepaths
    if title != cur_title:
        cur_title = title
        vidpath = title2vidpath[title]
        cur_pd_df = pd.read_csv(os.path.join(vidpath, 'preds', 'sent_biclass.csv'))
        cur_vid_framepaths = get_cur_vid_framepaths()

    preds = get_preds_from_df(cur_pd_df, window_len=int(window_len))
    data = {'preds': preds, 'framepaths': cur_vid_framepaths}

    return Response(
        json.dumps(data),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )

setup_initial_data()
