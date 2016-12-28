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

from natsort import natsorted

VIDEOS_PATH = 'data/videos/'

feast_path = os.path.join(VIDEOS_PATH, '@Animated/@NewDisney/Feast')
frames = natsorted([f for f in os.listdir(os.path.join(feast_path, 'frames')) if not f.startswith('.')])
frames = [os.path.join('@Animated/@NewDisney/Feast', 'frames', f) for f in frames]
data = {'frames': frames}

@app.route('/shape', methods=['GET'])
def shape():
    return render_template('plot_shape.html', data=json.dumps(data))
