# Run server
import os

from shape import app
app.run(port=int(os.environ.get("PORT", 7898)), debug=True)
