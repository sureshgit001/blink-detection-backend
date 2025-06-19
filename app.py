from flask import Flask
from flask_cors import CORS
from routes import configure_routes
import os
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Register routes
configure_routes(app)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
