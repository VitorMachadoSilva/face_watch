"""
FaceWatch — Face Recognition Service (Versão Otimizada)
REGRAS: 
 - Realtime: Strict Haar + Proporção Humana + Confiança Exponencial.
 - Treino: Cache de features + Normalização L2.
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

# Configurações de Extração
HOG_WIN   = (64, 64)
HOG_BLOCK = (16, 16)
HOG_STEP  = (8, 8)
HOG_CELL  = (8, 8)
HOG_BINS  = 9
HIST_BINS = 32

# Parâmetros Haar
REALTIME_HAAR_PARAMS = [(6, 100), (5, 80)]
CAPTURE_HAAR_PARAMS = [(3, 30), (2, 20)]
MIN_FACE_AREA_RATIO  = 0.04 

class FaceService:
    _model       = None
    _labels      = []
    _threshold   = None
    _train_dists = {}
    _cascades    = None
    _feat_cache  = {}

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
                logger.info(f"[FaceService] Modelo carregado: {len(set(cls._labels))} pessoas.")
            except Exception as e:
                logger.warning(f"[FaceService] Erro ao carregar: {e}")

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
        features, labels = [], []
        new_count = 0

        print("\n[TREINO] Iniciando extração otimizada...")
        for p in persons_with_photos:
            pid, name, fpath = p["id"], p["name"], p["face_photo"]

            if not fpath or not os.path.exists(fpath):
                if pid in cls._feat_cache:
                    features.append(cls._feat_cache[pid][1])
                    labels.append(pid)
                continue

            mtime = os.path.getmtime(fpath)
            cache_key = (pid, round(mtime))
            
            if pid in cls._feat_cache and cls._feat_cache[pid][0] == cache_key:
                features.append(cls._feat_cache[pid][1])
                labels.append(pid)
                continue

            feat = cls._file_to_feat(fpath)
            if feat is not None:
                cls._feat_cache[pid] = (cache_key, feat)
                features.append(feat)
                labels.append(pid)
                new_count += 1

        if not features:
            return False

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)

        from sklearn.neighbors import KNeighborsClassifier
        # Mudança para metric='euclidean' em dados L2-norm equivale à similaridade de cosseno
        clf = KNeighborsClassifier(n_neighbors=min(5, len(X)), metric="euclidean", weights="distance")
        clf.fit(X, y)

        # Calibragem de Threshold
        intra = []
        for lbl in set(y):
            idx = np.where(y == lbl)[0]
            if len(idx) < 2: continue
            for i in idx:
                others = X[np.setdiff1d(idx, [i])]
                intra.append(np.linalg.norm(X[i] - others, axis=1).min())

        cls._threshold = float(np.mean(intra) + max(np.std(intra) * 3, 0.15)) if intra else 0.8
        cls._model, cls._labels = clf, list(y)
        cls.save_model(model_path)
        print(f"[TREINO] Finalizado. Threshold: {cls._threshold:.4f}")
        return True

    @classmethod
    def identify_b64(cls, b64_str: str, require_face: bool = False) -> dict:
        if cls._model is None: return _resp("no_model")
        try:
            arr = cls._b64_to_array(b64_str)
        except: return _resp("feature_error")

        cv2 = _cv()
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        roi = cls._detect_face_strict(bgr) if require_face else cls._detect_face_loose(bgr)

        if roi is None: return _resp("no_face")
        
        feat = cls._roi_to_feat(roi)
        return cls._classify(feat) if feat is not None else _resp("feature_error")

    @classmethod
    def _classify(cls, feat: np.ndarray) -> dict:
        n = min(5, len(cls._labels))
        X = feat.reshape(1, -1)
        dists, idxs = cls._model.kneighbors(X, n_neighbors=n)

        votes = {}
        for i in range(n):
            lbl = int(cls._labels[idxs[0][i]])
            w = 1.0 / (float(dists[0][i])**2 + 1e-9)
            votes[lbl] = votes.get(lbl, 0.0) + w

        best_lbl = max(votes, key=votes.get)
        best_dist = min([float(dists[0][i]) for i in range(n) if int(cls._labels[idxs[0][i]]) == best_lbl])

        # Confiança Exponencial: Mais rigorosa que a linear
        conf_raw = np.exp(-1.5 * (best_dist / cls._threshold))
        conf = float(np.clip(conf_raw * 100, 0.0, 100.0))

        status = "identified" if best_dist <= cls._threshold else "unknown"
        return _resp(status, person_id=best_lbl if status=="identified" else None, 
                     confidence=round(conf, 1), distance=round(best_dist, 4))

    @classmethod
    def _detect_face_strict(cls, bgr: np.ndarray):
        cv2 = _cv(); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = bgr.shape[:2]; image_area = w * h

        for cascade in cls._get_cascades():
            for (min_n, min_sz) in REALTIME_HAAR_PARAMS:
                faces = cascade.detectMultiScale(gray, 1.1, min_n, minSize=(min_sz, min_sz))
                if len(faces) == 0: continue
                
                faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                for (x, y, fw, fh) in faces:
                    ratio = fw / fh
                    # Filtro de proporção: Foco em rostos humanos (verticais)
                    if not (0.6 <= ratio <= 1.0): continue
                    if (fw * fh) < (image_area * MIN_FACE_AREA_RATIO): continue
                    
                    pad = int(fh * 0.15)
                    return bgr[max(0, y-pad):min(h, y+fh+pad), max(0, x-pad):min(w, x+fw+pad)]
        return None

    @classmethod
    def _detect_face_loose(cls, bgr: np.ndarray):
        cv2 = _cv(); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = bgr.shape[:2]
        
        # 1. Tentativa com Haar Cascades (Deteção Real)
        for cascade in cls._get_cascades():
            for (min_n, min_sz) in CAPTURE_HAAR_PARAMS:
                faces = cascade.detectMultiScale(gray, 1.1, min_n, minSize=(min_sz, min_sz))
                if len(faces) > 0:
                    x, y, fw, fh = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
                    
                    # CORTE PRECISO: Pegamos exatamente o rosto com 5% de margem apenas
                    # Isso foca os gradientes nos olhos/nariz/boca e ignora o fundo
                    p = 0.05 
                    y1, y2 = max(0, y-int(fh*p)), min(h, y+fh+int(fh*p))
                    x1, x2 = max(0, x-int(fw*p)), min(w, x+fw+int(fw*p))
                    return bgr[y1:y2, x1:x2]
        
        # 2. SE FALHAR (Caso dos JPEGs ruins): 
        # Em vez de pegar a imagem toda (que traz o fundo), fazemos um corte 
        # fixo na área onde o rosto costuma estar em fotos de identificação.
        # Isso isola o rosto e descarta 60% do "ruído" da foto.
        return bgr[int(h*0.15):int(h*0.75), int(w*0.2):int(w*0.8)]
        


    @classmethod
    def _roi_to_feat(cls, roi: np.ndarray):
        if roi is None or roi.size == 0: return None
        cv2 = _cv()
        resized = cv2.resize(roi, HOG_WIN, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # CLAHE para normalizar iluminação
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # HOG (Estrutura)
        hog = cv2.HOGDescriptor(HOG_WIN, HOG_BLOCK, HOG_STEP, HOG_CELL, HOG_BINS)
        hog_feat = hog.compute(gray_eq).flatten()

        # Histograma (Distribuição de tom) - Peso reduzido (0.3)
        hist = cv2.calcHist([gray_eq], [0], None, [HIST_BINS], [0, 256]).flatten()
        hist = (hist / (hist.sum() + 1e-7)) * 0.3

        feat = np.concatenate([hog_feat, hist]).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 1e-7 else feat

    @classmethod
    def _b64_to_array(cls, b64_str: str) -> np.ndarray:
        if "," in b64_str: b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        Image, ImageOps = _pil()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        return np.array(img)

    @classmethod
    def _file_to_feat(cls, filepath: str):
        try:
            Image, ImageOps = _pil()
            img = ImageOps.exif_transpose(Image.open(filepath)).convert("RGB")
            bgr = _cv().cvtColor(np.array(img), _cv().COLOR_RGB2BGR)
            return cls._roi_to_feat(cls._detect_face_loose(bgr))
        except: return None

    @classmethod
    def _get_cascades(cls):
        if cls._cascades: return cls._cascades
        cv2 = _cv(); d = cv2.data.haarcascades
        cls._cascades = [cv2.CascadeClassifier(os.path.join(d, n)) for n in 
                         ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]
                         if os.path.exists(os.path.join(d, n))]
        return cls._cascades

def _resp(status, person_id=None, confidence=0.0, distance=None):
    d = {"status": status, "person_id": person_id, "confidence": confidence}
    if distance is not None: d["distance"] = distance
    return d