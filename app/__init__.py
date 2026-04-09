from flask import Flask
import os


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="../static")

    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    from app.routes import bp

    app.register_blueprint(bp)

    return app
