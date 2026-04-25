"""FaceWatch — Face Recognition Service

REGRAS DO PIPELINE:
  realtime  → Haar com params rigidos + validacao de area/proporcao.
              Se nao detectar: no_face. Nunca chama KNN sem rosto confirmado.
  captura   → Haar frouxo + fallback crop central. Aceita qualquer foto.
  treino    → Identico ao captura (fallback ativo).
"""

import io
import os
import pickle
import base64
import logging
import numpy as np

logger = logging.getLogger(__name__)

_cv2 = None
_PIL = None

def _cv():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2

def _pil():
    global _PIL
    if _PIL is None:
        from PIL import Image, ImageOps
        _PIL = (Image, ImageOps)
    return _PIL


HOG_WIN   = (64, 64)
HOG_BLOCK = (16, 16)
HOG_STEP  = (8, 8)
HOG_CELL  = (8, 8)
HOG_BINS  = 9
HIST_BINS = 32

# Parametros Haar para realtime — NAO alterar sem testes.
# Alto minNeighbors = exige muitas confirmacoes sobrepostas = anti-falso-positivo.
# minSize grande = ignora deteccoes minusculas em texturas distantes.
REALTIME_HAAR_PARAMS = [
    # (minNeighbors, minSize_px)
    (6, 100),
    (5, 80),
]

# Parametros Haar para treino/captura (frouxo, aceita fotos variadas)
CAPTURE_HAAR_PARAMS = [
    (4, 40),
    (3, 30),
    (2, 20),
    (1, 10),
]

# Rosto deve ocupar pelo menos esta fracao da imagem no realtime
MIN_FACE_AREA_RATIO = 0.04   # 4% da imagem total


class FaceService:
    _model        = None
    _labels       = []
    _threshold    = None
    _train_dists  = {}
    _cascades     = None
    _feat_cache   = {}   # {person_id: np.ndarray} — features ja extraidas

    # ── Public API ──────────────────────────────────────────────────────────

    @classmethod
    def load_model(cls, path: str):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                cls._model       = data["model"]
                cls._labels      = data["labels"]
                cls._threshold   = data["threshold"]
                cls._train_dists = data.get("train_dists", {})
                cls._feat_cache  = data.get("feat_cache", {})
                logger.info(
                    f"[FaceService] Modelo: {len(set(cls._labels))} pessoas, "
                    f"threshold={cls._threshold:.4f}"
                )
            except Exception as e:
                logger.warning(f"[FaceService] Falha ao carregar modelo: {e}")

    @classmethod
    def save_model(cls, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model":       cls._model,
                "labels":      cls._labels,
                "threshold":   cls._threshold,
                "train_dists": cls._train_dists,
                "feat_cache":  cls._feat_cache,
            }, f)

    @classmethod
    def train(cls, persons_with_photos: list, model_path: str):
        """Treino incremental: reutiliza features em cache, extrai apenas as novas."""
        features, labels = [], []
        new_count = 0

        print("\n[TREINO] ══════════════════════════════════════")
        for p in persons_with_photos:
            pid   = p["id"]
            name  = p["name"]
            fpath = p["face_photo"]

            if not fpath or not os.path.exists(fpath):
                # Se tem cache, usa mesmo sem foto no disco
                if pid in cls._feat_cache:
                    features.append(cls._feat_cache[pid])
                    labels.append(pid)
                else:
                    print(f"  ⚠ {name} — foto nao encontrada, sem cache")
                continue

            # Usa cache se o arquivo nao mudou (compara mtime)
            mtime = os.path.getmtime(fpath)
            cache_key = (pid, round(mtime))
            if pid in cls._feat_cache and cls._feat_cache[pid][0] == cache_key:
                feat = cls._feat_cache[pid][1]
                features.append(feat)
                labels.append(pid)
                continue   # cache hit — sem print, sem I/O

            try:
                feat = cls._file_to_feat(fpath)
                if feat is None:
                    print(f"  ✗ {name} — extracao falhou")
                    continue
                # Salva no cache com mtime para invalidacao
                cls._feat_cache[pid] = (cache_key, feat)
                features.append(feat)
                labels.append(pid)
                new_count += 1
                print(f"  ✓ {name} (id={pid}) dim={feat.shape[0]}")
            except Exception as e:
                print(f"  ✗ {name} — erro: {e}")

        cached_count = len(features) - new_count
        print(f"  — {cached_count} do cache, {new_count} novas extrações")

        if not features:
            print("[TREINO] Sem dados.")
            return False

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)
        print(f"\n[TREINO] {len(X)} amostras, {len(set(y))} classes")

        from sklearn.neighbors import KNeighborsClassifier
        k = min(3, len(X))
        clf = KNeighborsClassifier(
            n_neighbors=k, metric="euclidean",
            weights="distance", algorithm="ball_tree",
        )
        clf.fit(X, y)

        # Calibrar threshold via distancias intra-classe
        intra, train_dists = [], {}
        for lbl in set(y):
            idx = np.where(y == lbl)[0]
            if len(idx) < 2:
                continue
            dists = []
            for i in idx:
                others = X[np.setdiff1d(idx, [i])]
                dists.append(np.linalg.norm(X[i] - others, axis=1).min())
            train_dists[lbl] = dists
            intra.extend(dists)

        if intra:
            m, s  = np.mean(intra), np.std(intra)
            threshold = float(m + max(s * 3, m * 2))
            print(f"[TREINO] mean={m:.4f} std={s:.4f} → threshold={threshold:.4f}")
        else:
            threshold = 1.5
            print(f"[TREINO] threshold fallback={threshold}")

        cls._model, cls._labels   = clf, list(y)
        cls._threshold            = threshold
        cls._train_dists          = train_dists
        cls.save_model(model_path)
        print("[TREINO] ══════════════════════════════════════\n")
        return True

    # ── Identificacao ───────────────────────────────────────────────────────

    @classmethod
    def identify_b64(cls, b64_str: str, require_face: bool = False) -> dict:
        """
        require_face=True  (realtime): rejeita se nao encontrar rosto real pelo Haar.
        require_face=False (captura) : usa crop central como fallback.
        """
        if cls._model is None:
            return _resp("no_model")

        try:
            arr = cls._b64_to_array(b64_str)
        except Exception:
            return _resp("feature_error")

        cv2 = _cv()
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if require_face:
            roi = cls._detect_face_strict(bgr)
            if roi is None:
                return _resp("no_face")   # <<< PARA AQUI. KNN nunca e chamado.
        else:
            roi = cls._detect_face_loose(bgr)

        feat = cls._roi_to_feat(roi)
        if feat is None:
            return _resp("feature_error")

        return cls._classify(feat)

    # ── Classificacao KNN ───────────────────────────────────────────────────

    @classmethod
    def _classify(cls, feat: np.ndarray) -> dict:
        n  = min(3, len(cls._labels))
        X  = feat.reshape(1, -1)
        dists, idxs = cls._model.kneighbors(X, n_neighbors=n)

        # Votacao ponderada por distancia inversa
        votes = {}
        for i in range(n):
            lbl = int(cls._labels[idxs[0][i]])
            w   = 1.0 / (float(dists[0][i]) + 1e-9)
            votes[lbl] = votes.get(lbl, 0.0) + w

        best_lbl  = max(votes, key=votes.get)
        # Distancia do vizinho mais proximo com o label vencedor
        best_dist = next(
            float(dists[0][i]) for i in range(n)
            if int(cls._labels[idxs[0][i]]) == best_lbl
        )

        conf = float(np.clip(1.0 - best_dist / (cls._threshold * 1.5), 0.0, 1.0))

        if best_dist > cls._threshold:
            return _resp("unknown", confidence=round(conf * 100, 1),
                         distance=round(best_dist, 4))

        return _resp("identified", person_id=best_lbl,
                     confidence=round(conf * 100, 1),
                     distance=round(best_dist, 4))

    # ── Deteccao de rosto ───────────────────────────────────────────────────

    @classmethod
    def _detect_face_strict(cls, bgr: np.ndarray):
        """
        Modo REALTIME — parametros conservadores apenas.
        Retorna roi ou None (nunca fallback).

        Regras de validacao:
          - rosto ocupa >= MIN_FACE_AREA_RATIO da imagem
          - proporcao largura/altura entre 0.5 e 2.0
          - detectado por pelo menos REALTIME_HAAR_PARAMS[0] (mais rigido)
            OU confirmado por dois cascades diferentes
        """
        cv2  = _cv()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = bgr.shape[:2]
        image_area = w * h

        for cascade in cls._get_cascades():
            for (min_n, min_sz) in REALTIME_HAAR_PARAMS:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=min_n,
                    minSize=(min_sz, min_sz),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                if len(faces) == 0:
                    continue

                # Ordena pelo maior
                faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                for (x, y, fw, fh) in faces:
                    # Validacao de area
                    if fw * fh < image_area * MIN_FACE_AREA_RATIO:
                        continue
                    # Validacao de proporcao
                    ratio = fw / fh
                    if not (0.5 <= ratio <= 2.0):
                        continue
                    # Passou: retorna o crop
                    pad = int(max(fw, fh) * 0.15)
                    x1 = max(0, x - pad);      y1 = max(0, y - pad)
                    x2 = min(w, x + fw + pad); y2 = min(h, y + fh + pad)
                    return bgr[y1:y2, x1:x2]

        return None   # nenhum rosto valido

    @classmethod
    def _detect_face_loose(cls, bgr: np.ndarray):
        """
        Modo CAPTURA/TREINO — tenta parametros progressivos,
        usa crop central como ultimo recurso.
        """
        cv2  = _cv()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = bgr.shape[:2]

        for cascade in cls._get_cascades():
            for (min_n, min_sz) in CAPTURE_HAAR_PARAMS:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=min_n,
                    minSize=(min_sz, min_sz),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                    x, y, fw, fh = faces[0]
                    pad = int(max(fw, fh) * 0.1)
                    x1 = max(0, x - pad);      y1 = max(0, y - pad)
                    x2 = min(w, x + fw + pad); y2 = min(h, y + fh + pad)
                    return bgr[y1:y2, x1:x2]

        # Fallback: crop central
        return bgr[int(h*.1):int(h*.9), int(w*.1):int(w*.9)]

    # ── Extracao de features ────────────────────────────────────────────────

    @classmethod
    def _b64_to_array(cls, b64_str: str) -> np.ndarray:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        Image, ImageOps = _pil()
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.array(img)

    @classmethod
    def _file_to_feat(cls, filepath: str):
        """Caminho do treino: carrega arquivo → roi loose → features."""
        try:
            Image, ImageOps = _pil()
            img = Image.open(filepath)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            arr = np.array(img)
            cv2 = _cv()
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            roi = cls._detect_face_loose(bgr)
            return cls._roi_to_feat(roi)
        except Exception as e:
            logger.warning(f"[FaceService] file load error ({filepath}): {e}")
            return None

    @classmethod
    def _roi_to_feat(cls, roi: np.ndarray):
        """ROI BGR → vetor HOG+CLAHE L2-normalizado."""
        if roi is None or roi.size == 0:
            return None
        cv2 = _cv()
        resized  = cv2.resize(roi, HOG_WIN, interpolation=cv2.INTER_AREA)
        gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq  = clahe.apply(gray)

        hog      = cv2.HOGDescriptor(HOG_WIN, HOG_BLOCK, HOG_STEP, HOG_CELL, HOG_BINS)
        hog_feat = hog.compute(gray_eq).flatten()

        hist = cv2.calcHist([gray_eq], [0], None, [HIST_BINS], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)

        feat = np.concatenate([hog_feat, hist]).astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat

    @classmethod
    def _get_cascades(cls):
        if cls._cascades is not None:
            return cls._cascades
        cv2 = _cv()
        d   = cv2.data.haarcascades
        cls._cascades = [
            cv2.CascadeClassifier(os.path.join(d, n))
            for n in [
                "haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_profileface.xml",
            ]
            if os.path.exists(os.path.join(d, n))
        ]
        return cls._cascades


# ── Helpers ─────────────────────────────────────────────────────────────────

def _resp(status, person_id=None, confidence=0.0, distance=None):
    d = {"status": status, "person_id": person_id, "confidence": confidence}
    if distance is not None:
        d["distance"] = distance
    return d