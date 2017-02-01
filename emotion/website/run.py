# Run server
import os
import sys

from shape import app

if __name__ == '__main__':
    # import logging
    #
    # gunicorn_error_logger = logging.getLogger('gunicorn.error')
    # app.logger.handlers.extend(gunicorn_error_logger.handlers)
    # app.logger.setLevel(logging.DEBUG)
    # app.logger.debug('this will show in the log')

    app.run(port=int(os.environ.get("PORT", 7898)), debug=True)
    # app.run(host='0.0.0.0', port=7898)