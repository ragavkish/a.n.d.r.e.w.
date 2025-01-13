from flask import Flask, redirect, url_for, session, flash
from flask_mail import Mail, Message
from flask_dance.contrib.google import make_google_blueprint, google
from server_side.routes import init_routes
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("APP_SECRET_KEY", "default-secret-key")

    app.config["GOOGLE_OAUTH_CLIENT_ID"] = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    app.config["GOOGLE_OAUTH_CLIENT_SECRET"] = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    google_bp = make_google_blueprint(
        client_id=app.config["GOOGLE_OAUTH_CLIENT_ID"],
        client_secret=app.config["GOOGLE_OAUTH_CLIENT_SECRET"],
        redirect_to="google_login_callback",
    )
    app.register_blueprint(google_bp, url_prefix="/google_login")

    app.config["MAIL_SERVER"] = "smtp.gmail.com"
    app.config["MAIL_PORT"] = 587
    app.config["MAIL_USE_TLS"] = True
    app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
    app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
    mail = Mail(app)

    init_routes(app, mail)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)