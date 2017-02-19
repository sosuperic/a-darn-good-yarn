# Run server

import argparse
import os

from shape import app
from shape import views

if __name__ == '__main__':

    # Attempt to print (log) errors when running gunicorn (on deployment)
    # import logging
    # gunicorn_error_logger = logging.getLogger('gunicorn.error')
    # app.logger.handlers.extend(gunicorn_error_logger.handlers)
    # app.logger.setLevel(logging.DEBUG)
    # app.logger.debug('this will show in the log')

    # TODO:
    # This only works locally with python run.py, does not work with gunicorn...
    # Unrelated to argparse. Just running the last line (setup_initial_data(load_clusters=False)) doesn't work
    # on gunicorn
    # parser = argparse.ArgumentParser(description='Run GUI')
    # parser.add_argument('--load_clusters', dest='load_clusters', action='store_true')
    # cmdline = parser.parse_args()
    # views.setup_initial_data(load_clusters=cmdline.load_clusters)
    # views.setup_initial_data(load_clusters=False)

    app.run(port=int(os.environ.get("PORT", 7898)), debug=True)