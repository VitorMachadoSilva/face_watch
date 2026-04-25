"""FaceWatch — App Factory"""

import os
from flask import Flask
from .models import db, bcrypt
from .routes.auth   import auth_bp
from .routes.api    import api_bp
from .routes.pages  import pages_bp
from .services.face import FaceService


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
    )

    # ── Config ──────────────────────────────────────────────
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app.config["SECRET_KEY"] = os.urandom(32).hex()
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'facewatch.db')}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static", "uploads")
    app.config["MODEL_PATH"] = os.path.join(BASE_DIR, "database", "knn_model.pkl")
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)

    # ── Extensions ──────────────────────────────────────────
    db.init_app(app)
    bcrypt.init_app(app)

    # ── Blueprints ──────────────────────────────────────────
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp,   url_prefix="/api")
    app.register_blueprint(pages_bp)

    # ── Init DB + seed ──────────────────────────────────────
    with app.app_context():
        db.create_all()
        _seed_users()
        # Load existing model
        FaceService.load_model(app.config["MODEL_PATH"])

    return app


def _seed_users():
    from .models import User
    defaults = [
        ("admin",    "admin123",    "admin"),
        ("policial", "policial123", "officer"),
    ]
    for uname, pwd, role in defaults:
        if not User.query.filter_by(username=uname).first():
            from .models import bcrypt as _bc
            u = User(username=uname, role=role)
            u.password_hash = _bc.generate_password_hash(pwd).decode()
            from .models import db as _db
            _db.session.add(u)
    from .models import db as _db
    _db.session.commit()
