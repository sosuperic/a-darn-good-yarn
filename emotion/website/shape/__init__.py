# Initialize Flask app

import os
from flask import Flask

app = Flask(__name__)

from shape import views