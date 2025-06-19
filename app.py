from flask import Flask
from flask_cors import CORS
from routes import configure_routes

app = Flask(__name__, static_folder='static', template_folder='templates')

# Enable CORS for all routes
CORS(app)

# Register routes
configure_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
