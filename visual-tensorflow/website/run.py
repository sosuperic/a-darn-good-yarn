# Run server
import os

from shape import app

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 7898)), debug=True)
    # app.run(host='0.0.0.0', port=7898)