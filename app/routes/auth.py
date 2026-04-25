"""FaceWatch — Auth Routes"""

from flask import Blueprint, request, jsonify, session, redirect, url_for
from ..models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    user = User.query.filter_by(username=data.get("username", "")).first()
    if user and user.check_password(data.get("password", "")):
        session["user_id"]   = user.id
        session["username"]  = user.username
        session["role"]      = user.role
        return jsonify({"ok": True, "user": user.to_dict()})
    return jsonify({"ok": False, "error": "Credenciais inválidas"}), 401


@auth_bp.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.route("/api/me")
def me():
    if "user_id" not in session:
        return jsonify({"ok": False}), 401
    user = User.query.get(session["user_id"])
    if not user:
        return jsonify({"ok": False}), 401
    return jsonify({"ok": True, "user": user.to_dict()})
