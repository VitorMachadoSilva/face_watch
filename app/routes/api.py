"""FaceWatch — API Routes"""

import os
import json
import uuid
import base64
from functools import wraps
from datetime import datetime
import zipfile
from werkzeug.utils import secure_filename

from flask import Blueprint, request, jsonify, session, current_app, send_from_directory, abort
from ..models import db, Person, Occurrence
from ..services.face import FaceService

api_bp = Blueprint("api", __name__)

# ── ROTA DE IMPORTAÇÃO EM MASSA ──────────────────────────────────────────────
@api_bp.route("/import/bulk", methods=["POST"])
def bulk_import():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Arquivo ZIP não enviado"}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.zip'):
        return jsonify({"status": "error", "message": "O arquivo precisa ser um .zip"}), 400

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    faces_dir = os.path.join(upload_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(file, 'r') as z:
            imported_count = 0
            
            for filepath in z.namelist():
                # 1. Ignora arquivos de sistema e pastas vazias
                if filepath.startswith('__') or filepath.endswith('/') or \
                   not filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # 2. Lógica de extração de nome melhorada:
                # Divide o caminho: ['pastaFace', 'Thiago', 'foto.jpeg']
                parts = [p for p in filepath.split('/') if p]
                
                if len(parts) >= 2:
                    # Se houver subpastas, o nome da pessoa é a pasta IMEDIATAMENTE acima do arquivo
                    name_raw = parts[-2].replace('_', ' ').strip()
                else:
                    # Se o arquivo estiver na raiz do zip, ignora ou usa regra padrão
                    continue

                # 3. Busca ou cria a pessoa
                person = Person.query.filter_by(name=name_raw).first()
                if not person:
                    person = Person(name=name_raw, status="active", risk_level="low")
                    db.session.add(person)
                    db.session.flush()

                # 4. Salva a imagem
                img_data = z.read(filepath)
                ext = os.path.splitext(filepath)[1]
                unique_name = f"face_{person.id}_{uuid.uuid4().hex[:6]}{ext}"
                save_path = os.path.join(faces_dir, unique_name)
                
                with open(save_path, "wb") as f:
                    f.write(img_data)
                
                person.face_photo = f"faces/{unique_name}"
                imported_count += 1
            
            db.session.commit()
            
            # Retreina o modelo
            model_path = os.path.join(current_app.instance_path, "face_model.pkl")
            _train_model(model_path)
            
            return jsonify({
                "status": "success", 
                "message": f"Importação finalizada: {imported_count} rostos processados de subpastas."
            })
            
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Erro: {str(e)}"}), 500




# ── Auth guard ───────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Não autenticado"}), 401
        return f(*args, **kwargs)
    return decorated


# ── Stats ─────────────────────────────────────────────────────────────────────
@api_bp.route("/stats")
@login_required
def stats():
    total     = Person.query.count()
    high_risk = Person.query.filter_by(risk_level="high").count()
    fugitives = Person.query.filter_by(status="fugitive").count()
    detained  = Person.query.filter_by(status="detained").count()
    model_ok  = FaceService._model is not None
    return jsonify({
        "total_persons": total,
        "high_risk":     high_risk,
        "fugitives":     fugitives,
        "detained":      detained,
        "model_ready":   model_ok,
        "trained_labels": len(set(FaceService._labels)) if FaceService._labels else 0,
    })


# ── Persons CRUD ──────────────────────────────────────────────────────────────
@api_bp.route("/persons", methods=["GET"])
@login_required
def list_persons():
    q        = request.args.get("q", "").strip()
    risk     = request.args.get("risk", "")
    status   = request.args.get("status", "")
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))

    query = Person.query
    if q:
        like = f"%{q}%"
        query = query.filter(
            db.or_(Person.name.ilike(like), Person.nickname.ilike(like),
                   Person.address.ilike(like))
        )
    if risk:
        query = query.filter_by(risk_level=risk)
    if status:
        query = query.filter_by(status=status)

    query = query.order_by(Person.updated_at.desc())
    pag   = query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({
        "persons":  [p.to_dict() for p in pag.items],
        "total":    pag.total,
        "pages":    pag.pages,
        "page":     page,
    })


@api_bp.route("/persons", methods=["POST"])
@login_required
def create_person():
    data = request.get_json(silent=True) or {}
    p = Person(
        name            = data.get("name", "").strip(),
        nickname        = data.get("nickname", ""),
        gender          = data.get("gender", ""),
        age             = data.get("age"),
        skin_color      = data.get("skin_color", ""),
        height_cm       = data.get("height_cm"),
        address         = data.get("address", ""),
        frequent_places = json.dumps(data.get("frequent_places", [])),
        substances      = json.dumps(data.get("substances", [])),
        tattoos         = data.get("tattoos", ""),
        physical_marks  = data.get("physical_marks", ""),
        observations    = data.get("observations", ""),
        risk_level      = data.get("risk_level", "low"),
        status          = data.get("status", "active"),
    )
    if not p.name:
        return jsonify({"error": "Nome é obrigatório"}), 400
    db.session.add(p)
    db.session.commit()
    return jsonify(p.to_dict()), 201


@api_bp.route("/persons/<int:pid>", methods=["GET"])
@login_required
def get_person(pid):
    p = Person.query.get_or_404(pid)
    return jsonify(p.to_dict(full=True))


@api_bp.route("/persons/<int:pid>", methods=["PUT"])
@login_required
def update_person(pid):
    p    = Person.query.get_or_404(pid)
    data = request.get_json(silent=True) or {}
    fields = ["name","nickname","gender","age","skin_color","height_cm",
              "address","tattoos","physical_marks","observations","risk_level","status"]
    for f in fields:
        if f in data:
            setattr(p, f, data[f])
    if "frequent_places" in data:
        p.frequent_places = json.dumps(data["frequent_places"])
    if "substances" in data:
        p.substances = json.dumps(data["substances"])
    p.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify(p.to_dict())


@api_bp.route("/persons/<int:pid>", methods=["DELETE"])
@login_required
def delete_person(pid):
    p = Person.query.get_or_404(pid)
    db.session.delete(p)
    db.session.commit()
    _retrain(background=False)
    return jsonify({"ok": True})


# ── Photo upload ──────────────────────────────────────────────────────────────
@api_bp.route("/persons/<int:pid>/photo/face", methods=["POST"])
@login_required
def upload_face(pid):
    p = Person.query.get_or_404(pid)
    data = request.get_json(silent=True) or {}
    b64  = data.get("image_b64", "")
    if not b64:
        return jsonify({"error": "Imagem ausente"}), 400

    path = _save_b64_image(b64, f"face_{pid}", "faces")
    if not path:
        return jsonify({"error": "Falha ao salvar imagem"}), 500

    p.face_photo = path
    p.updated_at = datetime.utcnow()
    db.session.commit()

    # Retreina em background — resposta imediata, treino nao bloqueia
    _retrain(background=True)
    return jsonify({"ok": True, "path": path})


@api_bp.route("/persons/<int:pid>/photo/extra", methods=["POST"])
@login_required
def upload_extra(pid):
    p    = Person.query.get_or_404(pid)
    data = request.get_json(silent=True) or {}
    b64  = data.get("image_b64", "")
    if not b64:
        return jsonify({"error": "Imagem ausente"}), 400

    path   = _save_b64_image(b64, f"extra_{pid}_{uuid.uuid4().hex[:6]}", "extras")
    extras = json.loads(p.extra_photos or "[]")
    extras.append(path)
    p.extra_photos = json.dumps(extras)
    p.updated_at   = datetime.utcnow()
    db.session.commit()
    return jsonify({"ok": True, "path": path})


# ── Occurrences ──────────────────────────────────────────────────────────────
@api_bp.route("/persons/<int:pid>/occurrences", methods=["POST"])
@login_required
def add_occurrence(pid):
    Person.query.get_or_404(pid)
    data = request.get_json(silent=True) or {}
    occ  = Occurrence(
        person_id     = pid,
        type          = data.get("type", ""),
        location      = data.get("location", ""),
        description   = data.get("description", ""),
        substances    = json.dumps(data.get("substances", [])),
        registered_by = session.get("username", ""),
    )
    db.session.add(occ)
    db.session.commit()
    return jsonify(occ.to_dict()), 201


# ── Recognition ──────────────────────────────────────────────────────────────
@api_bp.route("/recognize", methods=["POST"])
@login_required
def recognize():
    data   = request.get_json(silent=True) or {}
    b64    = data.get("image_b64", "")
    # Captura manual/arquivo: usa fallback crop mesmo sem rosto detectado
    result = FaceService.identify_b64(b64, require_face=False)
    return _enrich_result(result)


@api_bp.route("/recognize/realtime", methods=["POST"])
@login_required
def recognize_realtime():
    import logging
    log = logging.getLogger("facewatch.realtime")
    data   = request.get_json(silent=True) or {}
    b64    = data.get("image_b64", "")
    result = FaceService.identify_b64(b64, require_face=True)
    log.warning(f"[REALTIME] status={result['status']} conf={result.get('confidence')} dist={result.get('distance')}")
    return _enrich_result(result)


@api_bp.route("/debug/recognize", methods=["POST"])
@login_required
def debug_recognize():
    """Endpoint de diagnostico — retorna detalhes internos do pipeline."""
    import logging
    data = request.get_json(silent=True) or {}
    b64  = data.get("image_b64", "")

    if not b64:
        return jsonify({"error": "sem imagem"})

    # Testa deteccao com cada conjunto de parametros
    try:
        import base64 as _b64, io
        import numpy as np
        from PIL import Image, ImageOps
        import cv2

        raw = _b64.b64decode(b64.split(",",1)[-1])
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        arr = np.array(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = bgr.shape[:2]

        cascade_dir = cv2.data.haarcascades
        results = []
        for cname in ["haarcascade_frontalface_default.xml","haarcascade_frontalface_alt2.xml","haarcascade_profileface.xml"]:
            cpath = os.path.join(cascade_dir, cname)
            if not os.path.exists(cpath): continue
            cc = cv2.CascadeClassifier(cpath)
            for (mn, ms) in [(5,80),(4,60),(3,40),(2,20),(1,10)]:
                faces = cc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=mn, minSize=(ms,ms))
                if len(faces) > 0:
                    fx,fy,fw,fh = sorted(faces, key=lambda r:r[2]*r[3], reverse=True)[0]
                    area_pct = round(fw*fh/(w*h)*100, 1)
                    ratio    = round(fw/fh, 2)
                    results.append({
                        "cascade": cname.replace("haarcascade_","").replace(".xml",""),
                        "minNeighbors": mn, "minSize": ms,
                        "face_w": int(fw), "face_h": int(fh),
                        "area_pct": area_pct, "ratio": ratio,
                        "passes_filter": area_pct >= 3.0 and 0.5 <= ratio <= 2.0
                    })
                    break  # encontrou neste cascade, vai pro proximo

        # KNN raw
        knn_result = FaceService.identify_b64(b64, require_face=False)
        rt_result  = FaceService.identify_b64(b64, require_face=True)

        return jsonify({
            "image_size": {"w": w, "h": h},
            "threshold": FaceService._threshold,
            "haar_detections": results,
            "knn_no_face_check": knn_result,
            "knn_with_face_check": rt_result,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ── Retrain ──────────────────────────────────────────────────────────────────
@api_bp.route("/retrain", methods=["POST"])
@login_required
def retrain():
    ok = _retrain(background=False)
    return jsonify({"ok": ok})


# ── Helpers ──────────────────────────────────────────────────────────────────
def _retrain(background: bool = False):
    """Retreina o modelo KNN.

    background=True  → roda em thread separada, nao bloqueia a resposta HTTP.
                        Ideal para upload de foto (1 pessoa adicionada).
    background=False → roda sincrono, usado em /api/retrain manual e delete.
    """
    import threading

    def _do_train(app_ctx):
        with app_ctx:
            _run_train()

    if background:
        from flask import current_app
        ctx = current_app.app_context()
        t = threading.Thread(target=_do_train, args=(ctx,), daemon=True)
        t.start()
        return True   # retorna imediatamente
    else:
        return _run_train()


def _run_train():
    upload_dir  = current_app.config["UPLOAD_FOLDER"]
    model_path  = current_app.config["MODEL_PATH"]
    persons     = Person.query.all()
    data = []
    for p in persons:
        if p.face_photo:
            if os.path.isabs(p.face_photo):
                full = p.face_photo
            else:
                full = os.path.join(upload_dir, p.face_photo)
            data.append({"id": p.id, "name": p.name, "face_photo": full})
    if data:
        return FaceService.train(data, model_path)
    return False


def _enrich_result(result: dict):
    if result.get("person_id"):
        p = Person.query.get(result["person_id"])
        if p:
            result["person"] = p.to_dict()
    return jsonify(result)


@api_bp.route("/uploads/<path:filename>")
def serve_upload(filename):
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    return send_from_directory(upload_dir, filename)


def _save_b64_image(b64: str, name: str, subfolder: str):
    """Save base64 image. Returns path relative to UPLOAD_FOLDER (e.g. 'faces/face_1.jpg')."""
    try:
        upload_dir = current_app.config["UPLOAD_FOLDER"]
        subdir     = os.path.join(upload_dir, subfolder)
        os.makedirs(subdir, exist_ok=True)

        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw   = base64.b64decode(b64)
        fname = f"{name}.jpg"
        fpath = os.path.join(subdir, fname)
        with open(fpath, "wb") as f:
            f.write(raw)
        # Return relative to UPLOAD_FOLDER so serve_upload finds it
        return f"{subfolder}/{fname}"
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"_save_b64_image error: {e}")
        return None

def _train_model(model_path):
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    data = []
    
    # Busca todas as pessoas que têm uma foto vinculada
    persons = Person.query.filter(Person.face_photo != None).all()
    
    print(f"\n[DEBUG TREINO] Iniciando sincronização de {len(persons)} pessoas...")

    for p in persons:
        # Garante o caminho absoluto correto
        if os.path.isabs(p.face_photo):
            full_path = p.face_photo
        else:
            full_path = os.path.join(upload_dir, p.face_photo)

        if os.path.exists(full_path):
            data.append({"id": p.id, "name": p.name, "face_photo": full_path})
        else:
            print(f"[ERRO] Arquivo não encontrado para {p.name}: {full_path}")

    if data:
        print(f"[DEBUG TREINO] Enviando {len(data)} candidatos para o FaceService...")
        success = FaceService.train(data, model_path)
        
        # O FaceService._labels contém quem realmente foi treinado com sucesso
        trained_count = len(set(FaceService._labels)) if FaceService._labels else 0
        print(f"[DEBUG TREINO] Finalizado! Pessoas no modelo: {trained_count}")
        return success
    
    print("[AVISO] Nenhum dado válido para treinar.")
    return False