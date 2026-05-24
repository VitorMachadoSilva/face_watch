"""FaceWatch — Database Models"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

db     = SQLAlchemy()
bcrypt = Bcrypt()


class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20), default="officer")   # admin | officer
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {"id": self.id, "username": self.username, "role": self.role}


class Person(db.Model):
    __tablename__ = "persons"
    id              = db.Column(db.Integer, primary_key=True)
    name            = db.Column(db.String(120), nullable=False)
    nickname        = db.Column(db.String(80))
    cpf             = db.Column(db.String(14), nullable=True)   # formato: 000.000.000-00
    gender          = db.Column(db.String(20))
    age             = db.Column(db.Integer)
    skin_color      = db.Column(db.String(40))
    height_cm       = db.Column(db.Integer)
    address         = db.Column(db.Text)
    frequent_places = db.Column(db.Text)   # JSON array string
    substances      = db.Column(db.Text)   # JSON array string
    tattoos         = db.Column(db.Text)
    physical_marks  = db.Column(db.Text)
    observations    = db.Column(db.Text)
    risk_level      = db.Column(db.String(10), default="low")    # low | medium | high
    status          = db.Column(db.String(20), default="active") # active | detained | fugitive | released
    face_photo      = db.Column(db.String(256))   # relative path
    extra_photos    = db.Column(db.Text)           # JSON array of paths
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at      = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    occurrences = db.relationship("Occurrence", backref="person", lazy=True, cascade="all, delete-orphan")

    def to_dict(self, full=False):
        import json
        d = {
            "id":           self.id,
            "name":         self.name,
            "nickname":     self.nickname,
            "cpf":          self.cpf,
            "gender":       self.gender,
            "age":          self.age,
            "skin_color":   self.skin_color,
            "height_cm":    self.height_cm,
            "address":      self.address,
            "frequent_places": json.loads(self.frequent_places or "[]"),
            "substances":      json.loads(self.substances or "[]"),
            "tattoos":         self.tattoos,
            "physical_marks":  self.physical_marks,
            "observations":    self.observations,
            "risk_level":      self.risk_level,
            "status":          self.status,
            "face_photo":      self.face_photo,
            "extra_photos":    json.loads(self.extra_photos or "[]"),
            "created_at":      self.created_at.isoformat() if self.created_at else None,
            "updated_at":      self.updated_at.isoformat() if self.updated_at else None,
        }
        if full:
            d["occurrences"] = [o.to_dict() for o in self.occurrences]
        return d


class Occurrence(db.Model):
    __tablename__ = "occurrences"
    id          = db.Column(db.Integer, primary_key=True)
    person_id   = db.Column(db.Integer, db.ForeignKey("persons.id"), nullable=False)
    type        = db.Column(db.String(60))
    location    = db.Column(db.String(200))
    description = db.Column(db.Text)
    substances  = db.Column(db.Text)   # JSON array
    registered_by = db.Column(db.String(80))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        return {
            "id":          self.id,
            "person_id":   self.person_id,
            "type":        self.type,
            "location":    self.location,
            "description": self.description,
            "substances":  json.loads(self.substances or "[]"),
            "registered_by": self.registered_by,
            "created_at":  self.created_at.isoformat() if self.created_at else None,
        }