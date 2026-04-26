"""
FaceWatch — Face Recognition Service
Motor: facenet-pytorch (InceptionResnetV1 + MTCNN)
Treinado em VGGFace2 — reconhecimento robusto a distância, luz e ângulo.

Fluxo:
  Imagem → MTCNN detecta e alinha rosto → InceptionResnetV1 gera embedding 512d
  → Similaridade cosseno contra embeddings cadastrados → threshold calibrado

Fallback automático se facenet-pytorch não estiver instalado:
  → HOG+KNN (motor antigo) com aviso no terminal.
"""

import io, os, pickle, base64, logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Detecta se facenet-pytorch está disponível ───────────────────────────────
try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    _FACENET_OK = True
except ImportError:
    _FACENET_OK = False
    logger.warning("[FaceService] facenet-pytorch não encontrado — usando HOG+KNN (qualidade inferior).")
    logger.warning("[FaceService] Para instalar: pip install torch torchvision facenet-pytorch")


def _resp(status, person_id=None, confidence=0.0, distance=None):
    d = {"status": status, "person_id": person_id, "confidence": confidence}
    if distance is not None:
        d["distance"] = distance
    return d


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL — FaceNet (facenet-pytorch)
# ═══════════════════════════════════════════════════════════════════════════════

class FaceNetEngine:
    """
    Detecção: MTCNN (detecta + alinha o rosto, robusto a ângulo/luz)
    Embedding: InceptionResnetV1 pré-treinado em VGGFace2 (512 dimensões)
    Classificação: Similaridade cosseno + threshold por pessoa
    """

    _mtcnn   = None
    _resnet  = None
    _db      = {}        # {person_id: [embedding, embedding, ...]}
    _threshold = 0.6     # distância cosseno — calibrado após cadastro

    @classmethod
    def _load_models(cls):
        if cls._mtcnn is None:
            cls._mtcnn = MTCNN(
                image_size=160,
                margin=20,
                keep_all=False,
                select_largest=True,
                device='cpu',
                post_process=True,
            )
        if cls._resnet is None:
            cls._resnet = InceptionResnetV1(pretrained='vggface2').eval()

    @classmethod
    def get_embedding(cls, pil_img):
        """PIL Image → tensor 512d ou None se não detectar rosto."""
        cls._load_models()
        face_tensor = cls._mtcnn(pil_img)
        if face_tensor is None:
            return None
        with torch.no_grad():
            emb = cls._resnet(face_tensor.unsqueeze(0))
        return emb[0].cpu().numpy()   # (512,)

    @classmethod
    def get_embedding_loose(cls, pil_img):
        """Igual ao acima mas aceita imagem inteira se MTCNN falhar (cadastro)."""
        emb = cls.get_embedding(pil_img)
        if emb is not None:
            return emb

        # Fallback: crop central e tenta de novo
        cls._load_models()
        import PIL.Image as PILImage
        w, h = pil_img.size
        crop = pil_img.crop((int(w*0.1), int(h*0.05), int(w*0.9), int(h*0.85)))
        crop = crop.resize((160, 160), PILImage.LANCZOS)

        try:
            face_tensor = cls._mtcnn(crop)
            if face_tensor is None:
                # Força resize direto sem detecção
                import torchvision.transforms as T
                t = T.Compose([T.Resize((160,160)), T.ToTensor(),
                               T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
                face_tensor = t(crop)
            with torch.no_grad():
                emb = cls._resnet(face_tensor.unsqueeze(0))
            return emb[0].cpu().numpy()
        except Exception as e:
            logger.warning(f"[FaceNet] get_embedding_loose fallback falhou: {e}")
            return None

    @classmethod
    def cosine_distance(cls, a, b):
        """Distância cosseno entre dois vetores (0=idêntico, 2=oposto)."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))

    @classmethod
    def identify(cls, pil_img, strict=True):
        if not cls._db:
            return _resp("no_model")

        if strict:
            emb = cls.get_embedding(pil_img)
            if emb is None:
                return _resp("no_face")
        else:
            emb = cls.get_embedding_loose(pil_img)
            if emb is None:
                return _resp("feature_error")

        best_pid  = None
        best_dist = float("inf")

        for pid, embs in cls._db.items():
            dists = [cls.cosine_distance(emb, e) for e in embs]
            d_min = min(dists)
            if d_min < best_dist:
                best_dist = d_min
                best_pid  = pid

        # Log diagnóstico no terminal
        print(f"[IDENTIFY] dist={best_dist:.4f} threshold={cls._threshold:.4f} pid={best_pid}")

        conf = float(np.clip(1.0 - best_dist / max(cls._threshold * 1.5, 1e-6), 0.0, 1.0))

        if best_dist > cls._threshold:
            print(f"[IDENTIFY] → unknown (dist {best_dist:.4f} > threshold {cls._threshold:.4f})")
            return _resp("unknown",
                         confidence=round(conf * 100, 1),
                         distance=round(best_dist, 4))

        print(f"[IDENTIFY] → identified pid={best_pid} conf={conf*100:.1f}%")
        return _resp("identified",
                     person_id=int(best_pid),
                     confidence=round(conf * 100, 1),
                     distance=round(best_dist, 4))

    @classmethod
    def calibrate_threshold(cls):
        """
        Threshold baseado em intra-classe (mesma pessoa) e inter-classe (pessoas diferentes).
        Garante que a própria pessoa seja reconhecida antes de calibrar rejeição.
        """
        pids = list(cls._db.keys())

        # Distâncias intra-classe (mesma pessoa, fotos diferentes)
        intra = []
        for pid in pids:
            embs = cls._db[pid]
            for i in range(len(embs)):
                for j in range(i+1, len(embs)):
                    intra.append(cls.cosine_distance(embs[i], embs[j]))

        # Distâncias inter-classe (pessoas diferentes)
        inter = []
        for i in range(len(pids)):
            for j in range(i+1, len(pids)):
                for ea in cls._db[pids[i]]:
                    for eb in cls._db[pids[j]]:
                        inter.append(cls.cosine_distance(ea, eb))

        if intra and inter:
            # FaceNet VGGFace2: distancias intra tipicas 0.4-0.8, inter tipicas 0.9-1.4
            # Threshold no ponto medio entre pior intra e melhor inter
            intra_max = float(np.percentile(intra, 95))
            inter_min = float(np.percentile(inter, 5))
            if inter_min > intra_max:
                mid = (intra_max + inter_min) / 2.0
            else:
                # Overlap entre intra e inter — usa intra_max + margem
                mid = intra_max * 1.15
            cls._threshold = float(np.clip(mid, 0.70, 1.20))
            print(f"[TREINO] Threshold={cls._threshold:.4f} "
                  f"(intra_max={intra_max:.4f} inter_min={inter_min:.4f})")
        elif inter:
            # Sem dados intra — usa 80% da distancia minima inter
            cls._threshold = float(np.clip(np.percentile(inter, 20) * 0.85, 0.70, 1.10))
            print(f"[TREINO] Threshold={cls._threshold:.4f} (só inter-classe)")
        elif intra:
            # Só 1 pessoa — threshold bem generoso: pior intra * 1.5
            intra_max = float(np.percentile(intra, 95))
            cls._threshold = float(np.clip(intra_max * 1.5, 0.75, 1.20))
            print(f"[TREINO] Threshold={cls._threshold:.4f} (1 pessoa, intra_max={intra_max:.4f})")
        else:
            # 1 pessoa, 1 foto — threshold fixo baseado no comportamento real do FaceNet
            cls._threshold = 0.90
            print(f"[TREINO] Threshold={cls._threshold:.4f} (fallback FaceNet)")


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR FALLBACK — HOG + KNN (quando facenet-pytorch não disponível)
# ═══════════════════════════════════════════════════════════════════════════════

class HOGEngine:
    """Motor legado — mantido como fallback."""

    HOG_WIN = (64, 64); HOG_BLOCK = (16, 16)
    HOG_STEP = (8, 8);  HOG_CELL  = (8, 8); HOG_BINS = 9; HIST_BINS = 32

    _model = None; _labels = []; _threshold = 0.55; _feat_cache = {}
    _cascades = None

    @classmethod
    def _cv(cls):
        import cv2; return cv2

    @classmethod
    def _get_cascades(cls):
        if cls._cascades: return cls._cascades
        cv2 = cls._cv(); d = cv2.data.haarcascades
        cls._cascades = [cv2.CascadeClassifier(os.path.join(d, n)) for n in
                         ["haarcascade_frontalface_default.xml",
                          "haarcascade_frontalface_alt2.xml"]
                         if os.path.exists(os.path.join(d, n))]
        return cls._cascades

    @classmethod
    def _detect(cls, bgr, strict=False):
        cv2 = cls._cv()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = bgr.shape[:2]
        params = [(5,80),(4,60)] if strict else [(3,40),(2,20),(1,10)]
        for cascade in cls._get_cascades():
            for mn, ms in params:
                faces = cascade.detectMultiScale(gray, 1.05, mn, minSize=(ms,ms))
                if len(faces) > 0:
                    x, y, fw, fh = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
                    if strict and (fw*fh)/(w*h) < 0.03: continue
                    p = int(max(fw,fh)*0.1)
                    return bgr[max(0,y-p):min(h,y+fh+p), max(0,x-p):min(w,x+fw+p)], True
        if strict: return None, False
        y1,y2 = int(h*.1), int(h*.9); x1,x2 = int(w*.1), int(w*.9)
        return bgr[y1:y2, x1:x2], False

    @classmethod
    def _feat(cls, roi):
        cv2 = cls._cv()
        if roi is None or roi.size == 0: return None
        r = cv2.resize(roi, cls.HOG_WIN, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        eq = cv2.createCLAHE(2.0,(8,8)).apply(g)
        hog = cv2.HOGDescriptor(cls.HOG_WIN,cls.HOG_BLOCK,cls.HOG_STEP,cls.HOG_CELL,cls.HOG_BINS)
        hf = hog.compute(eq).flatten()
        hist = cv2.calcHist([eq],[0],None,[cls.HIST_BINS],[0,256]).flatten()
        hist /= (hist.sum()+1e-7)
        feat = np.concatenate([hf, hist]).astype(np.float32)
        n = np.linalg.norm(feat)
        return feat/n if n > 1e-7 else feat

    @classmethod
    def identify(cls, pil_img, strict=True):
        if cls._model is None: return _resp("no_model")
        import cv2
        arr = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        roi, found = cls._detect(bgr, strict=strict)
        if strict and not found: return _resp("no_face")
        feat = cls._feat(roi)
        if feat is None: return _resp("feature_error")

        X = feat.reshape(1,-1)
        dists, idxs = cls._model.kneighbors(X, n_neighbors=min(3,len(cls._labels)))
        best_dist = float(dists[0][0])
        best_lbl  = int(cls._labels[idxs[0][0]])
        conf = float(np.clip(1.0 - best_dist/(cls._threshold*1.5), 0, 1))

        if best_dist > cls._threshold or conf < 0.75:
            return _resp("unknown", confidence=round(conf*100,1), distance=round(best_dist,4))
        return _resp("identified", person_id=best_lbl,
                     confidence=round(conf*100,1), distance=round(best_dist,4))


# ═══════════════════════════════════════════════════════════════════════════════
#  FACHADA PÚBLICA — FaceService
# ═══════════════════════════════════════════════════════════════════════════════

class FaceService:
    """
    Interface pública única. Usa FaceNetEngine se disponível, HOGEngine como fallback.
    """
    _model_path = None

    # ── Persistência ──────────────────────────────────────────────────────────

    @classmethod
    def load_model(cls, path: str):
        cls._model_path = path
        if not os.path.exists(path):
            return

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            engine = data.get("engine", "hog")

            if engine == "facenet" and _FACENET_OK:
                FaceNetEngine._db        = data["db"]
                FaceNetEngine._threshold = data["threshold"]
                n_persons = len(FaceNetEngine._db)
                n_photos  = sum(len(v) for v in FaceNetEngine._db.values())
                print(f"[FaceService] FaceNet: {n_persons} pessoas, {n_photos} fotos, "
                      f"threshold={FaceNetEngine._threshold:.4f}")
            else:
                HOGEngine._model     = data.get("model")
                HOGEngine._labels    = data.get("labels", [])
                HOGEngine._threshold = data.get("threshold", 0.55)
                HOGEngine._feat_cache= data.get("feat_cache", {})
                print(f"[FaceService] HOG: {len(set(HOGEngine._labels))} pessoas, "
                      f"threshold={HOGEngine._threshold:.4f}")
        except Exception as e:
            logger.warning(f"[FaceService] Erro ao carregar modelo: {e}")

    @classmethod
    def save_model(cls, path: str):
        if _FACENET_OK:
            with open(path, "wb") as f:
                pickle.dump({
                    "engine":    "facenet",
                    "db":        FaceNetEngine._db,
                    "threshold": FaceNetEngine._threshold,
                }, f)
        else:
            with open(path, "wb") as f:
                pickle.dump({
                    "engine":     "hog",
                    "model":      HOGEngine._model,
                    "labels":     HOGEngine._labels,
                    "threshold":  HOGEngine._threshold,
                    "feat_cache": HOGEngine._feat_cache,
                }, f)

    # ── Treino ────────────────────────────────────────────────────────────────

    @classmethod
    def train(cls, persons_with_photos: list, model_path: str):
        """
        persons_with_photos: [{"id": int, "name": str, "face_photo": str_path}, ...]
        Suporta múltiplas entradas do mesmo id (3-5 fotos por pessoa).
        """
        print(f"\n[TREINO] {'FaceNet (VGGFace2)' if _FACENET_OK else 'HOG+KNN (fallback)'}")
        print("═" * 50)

        if _FACENET_OK:
            return cls._train_facenet(persons_with_photos, model_path)
        else:
            return cls._train_hog(persons_with_photos, model_path)

    @classmethod
    def _train_facenet(cls, persons_with_photos, model_path):
        from PIL import Image, ImageOps

        # Agrupa fotos por pessoa
        person_photos = {}  # {pid: [(name, path), ...]}
        for p in persons_with_photos:
            pid = p["id"]
            if pid not in person_photos:
                person_photos[pid] = (p["name"], [])
            person_photos[pid][1].append(p["face_photo"])

        db = {}
        errors = 0

        for pid, (name, paths) in person_photos.items():
            embs = []
            for fpath in paths:
                if not fpath or not os.path.exists(fpath):
                    continue
                try:
                    img = ImageOps.exif_transpose(Image.open(fpath)).convert("RGB")
                    emb = FaceNetEngine.get_embedding_loose(img)
                    if emb is not None:
                        embs.append(emb)
                except Exception as e:
                    logger.warning(f"  ✗ {name} [{fpath}]: {e}")
                    errors += 1

            if embs:
                db[pid] = embs
                print(f"  ✓ {name} (id={pid}) — {len(embs)} foto(s)")
            else:
                print(f"  ✗ {name} (id={pid}) — nenhum embedding extraído")
                errors += 1

        if not db:
            print("[TREINO] Nenhum dado válido.")
            return False

        FaceNetEngine._db = db
        FaceNetEngine.calibrate_threshold()
        cls.save_model(model_path)

        print(f"\n[TREINO] Concluído: {len(db)} pessoas | erros: {errors}")
        print("═" * 50 + "\n")
        return True

    @classmethod
    def _train_hog(cls, persons_with_photos, model_path):
        import cv2
        from PIL import Image, ImageOps
        from sklearn.neighbors import KNeighborsClassifier

        features, labels, new_count = [], [], 0

        for p in persons_with_photos:
            pid, name, fpath = p["id"], p["name"], p["face_photo"]
            if not fpath or not os.path.exists(fpath):
                if pid in HOGEngine._feat_cache:
                    features.append(HOGEngine._feat_cache[pid][1])
                    labels.append(pid)
                continue

            mtime = round(os.path.getmtime(fpath))
            ck = (pid, mtime)
            if pid in HOGEngine._feat_cache and HOGEngine._feat_cache[pid][0] == ck:
                features.append(HOGEngine._feat_cache[pid][1])
                labels.append(pid)
                continue

            try:
                img = ImageOps.exif_transpose(Image.open(fpath)).convert("RGB")
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                roi, _ = HOGEngine._detect(bgr, strict=False)
                feat = HOGEngine._feat(roi)
                if feat is not None:
                    HOGEngine._feat_cache[pid] = (ck, feat)
                    features.append(feat); labels.append(pid); new_count += 1
                    print(f"  ✓ {name} (id={pid})")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

        if not features:
            return False

        X, y = np.array(features, dtype=np.float32), np.array(labels)
        clf = KNeighborsClassifier(n_neighbors=min(5,len(X)), metric="euclidean", weights="distance")
        clf.fit(X, y)

        intra = []
        for lbl in set(y):
            idx = np.where(y==lbl)[0]
            if len(idx) < 2: continue
            for i in idx:
                others = X[np.setdiff1d(idx,[i])]
                intra.append(np.linalg.norm(X[i]-others, axis=1).min())

        HOGEngine._threshold = float(np.mean(intra)+max(np.std(intra)*3,0.15)) if intra else 0.55
        HOGEngine._threshold = min(HOGEngine._threshold, 0.80)
        HOGEngine._model, HOGEngine._labels = clf, list(y)
        cls.save_model(model_path)
        print(f"[TREINO] HOG concluído. Threshold={HOGEngine._threshold:.4f}\n")
        return True

    # ── Identificação ─────────────────────────────────────────────────────────

    @classmethod
    def identify_b64(cls, b64_str: str, require_face: bool = False) -> dict:
        try:
            pil_img = cls._b64_to_pil(b64_str)
        except Exception as e:
            logger.warning(f"[FaceService] decode error: {e}")
            return _resp("feature_error")

        if _FACENET_OK:
            if not FaceNetEngine._db:
                return _resp("no_model")
            return FaceNetEngine.identify(pil_img, strict=require_face)
        else:
            return HOGEngine.identify(pil_img, strict=require_face)

    @classmethod
    def _b64_to_pil(cls, b64_str: str):
        from PIL import Image, ImageOps
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw))
        return ImageOps.exif_transpose(img).convert("RGB")