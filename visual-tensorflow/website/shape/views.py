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
import numpy
import pandas as pd


from natsort import natsorted

VIDEOS_PATH = 'data/videos/'

feast_path = os.path.join(VIDEOS_PATH, '@Animated/@NewDisney/Feast')
frames = natsorted([f for f in os.listdir(os.path.join(feast_path, 'frames')) if not f.startswith('.')])
frames = [os.path.join('@Animated/@NewDisney/Feast', 'frames', f) for f in frames]
pred = os.path.join('@Animated/@NewDisney/Feast', 'preds', 'sent_biclass_0.csv')


def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w=numpy.ones(window_len,'d')
    else:  
            w=eval('numpy.'+window+'(window_len)')
    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]


vid = pd.read_csv('data/videos/@Animated/@NewDisney/Feast/preds/sent_biclass_0.csv')
smoothed = smooth(vid.pos.values, window_len=int(48))

data = {'frames': frames, 'pred': pred, 'smoothed': list(smoothed)}


@app.route('/api/pred/<window>', methods=['GET'])
def get_preds(window):
    vid = pd.read_csv('data/videos/@Animated/@NewDisney/Feast/preds_old_torch/posneg.csv',
                  header=None, names=['Positive', 'Negative'])
    smoothed = smooth(vid.Positive.values, window_len=int(window))

    return Response(
        json.dumps(list(smoothed)),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/shape', methods=['GET'])
def shape():
    return render_template('plot_shape.html', data=json.dumps(data))
