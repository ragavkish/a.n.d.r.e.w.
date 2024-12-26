from flask import Flask
from flask_cors import CORS
from server_side.routes import init_routes

def create_app():
    app = Flask(__name__)
    CORS(app)

    init_routes(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)