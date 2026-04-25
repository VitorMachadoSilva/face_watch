"""FaceWatch — Page Routes (HTML views)"""

from flask import Blueprint, render_template, session, redirect, url_for

pages_bp = Blueprint("pages", __name__)


def _logged():
    return "user_id" in session




@pages_bp.route("/")
def index():
    if not _logged():
        return redirect(url_for("pages.login_page"))
    return redirect(url_for("pages.identify"))


@pages_bp.route("/login")
def login_page():
    if _logged():
        return redirect(url_for("pages.identify"))
    return render_template("login.html")


@pages_bp.route("/identify")
def identify():
    if not _logged():
        return redirect(url_for("pages.login_page"))
    return render_template("identify.html", user=session)


@pages_bp.route("/register")
def register():
    if not _logged():
        return redirect(url_for("pages.login_page"))
    return render_template("register.html", user=session)


@pages_bp.route("/search")
def search():
    if not _logged():
        return redirect(url_for("pages.login_page"))
    return render_template("search.html", user=session)


@pages_bp.route("/person/<int:person_id>")
def person_detail(person_id):
    if not _logged():
        return redirect(url_for("pages.login_page"))
    return render_template("person.html", user=session, person_id=person_id)


@pages_bp.route("/import")
def import_page():
    if not _logged():
        return redirect(url_for("pages.login_page"))
    return render_template("import.html", user=session)