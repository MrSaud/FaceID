import base64
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


MODEL_DIR = Path(__file__).resolve().parent / "ml_models"
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
MINIFAS_V2_PATH = MODEL_DIR / "MiniFASNetV2.onnx"
MINIFAS_V1SE_PATH = MODEL_DIR / "MiniFASNetV1SE.onnx"

# MiniFASNet class index 1 = real face; 0 and 2 = spoof (print / replay).
_ANTISPOOF_MODELS = (
    (MINIFAS_V2_PATH, 2.7),
    (MINIFAS_V1SE_PATH, 4.0),
)


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def _decode_image_bytes(data: bytes):
    if not data:
        raise ValueError("image payload is empty")
    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image payload is not a valid image")
    return image


def _decode_base64_to_bgr(image_base64):
    raw = str(image_base64 or "").strip()
    if not raw:
        raise ValueError("image_base64 missing")
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    # Strip newlines; restore '+' lost by form-urlencoded; accept base64url
    raw = raw.replace("\r", "").replace("\n", "").replace("\t", "")
    raw = raw.replace(" ", "+").replace("-", "+").replace("_", "/")
    pad = (-len(raw)) % 4
    if pad:
        raw += "=" * pad
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise ValueError("image_base64 decode failed") from exc
    return _decode_image_bytes(data)


def _has_sface_pipeline():
    return (
        YUNET_MODEL_PATH.exists()
        and SFACE_MODEL_PATH.exists()
        and hasattr(cv2, "FaceDetectorYN_create")
        and hasattr(cv2, "FaceRecognizerSF_create")
    )


def _create_yunet(image_w, image_h):
    return cv2.FaceDetectorYN_create(
        str(YUNET_MODEL_PATH),
        "",
        (int(image_w), int(image_h)),
        score_threshold=0.85,
        nms_threshold=0.3,
        top_k=500,
    )


def _face_row_to_dict(face):
    x, y, w, h = [float(v) for v in face[:4]]
    landmarks = [
        {"x": float(face[4]), "y": float(face[5])},
        {"x": float(face[6]), "y": float(face[7])},
        {"x": float(face[8]), "y": float(face[9])},
        {"x": float(face[10]), "y": float(face[11])},
        {"x": float(face[12]), "y": float(face[13])},
    ]
    return {
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "score": round(float(face[14]), 4),
        "landmarks": landmarks,
    }


def _detect_faces(image_bgr):
    if not _has_sface_pipeline():
        return None, "YuNet model not available"
    h, w = image_bgr.shape[:2]
    detector = _create_yunet(w, h)
    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)
    if faces is None or len(faces) == 0:
        return [], ""
    return [_face_row_to_dict(face) for face in faces], ""


def _detect_best_face(image_bgr):
    h, w = image_bgr.shape[:2]
    detector = _create_yunet(w, h)
    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)
    if faces is None or len(faces) == 0:
        return None
    best_idx = int(np.argmax(faces[:, 14]))
    return faces[best_idx]


def _extract_aligned_face(image_bgr):
    face = _detect_best_face(image_bgr)
    if face is None:
        return None, "no face detected"
    recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL_PATH), "")
    aligned = recognizer.alignCrop(image_bgr, face)
    if aligned is None or aligned.size == 0:
        return None, "failed to align face"
    return aligned, ""


def _softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _has_antispoof_pipeline():
    return (
        ort is not None
        and YUNET_MODEL_PATH.exists()
        and hasattr(cv2, "FaceDetectorYN_create")
        and any(path.exists() for path, _ in _ANTISPOOF_MODELS)
    )


@lru_cache(maxsize=4)
def _load_antispoof_session(model_path: str):
    return ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )


def _crop_face_for_antispoof(image_bgr, bbox_xywh, scale):
    """Expanded crop around YuNet bbox (Silent-Face / MiniFASNet convention)."""
    src_h, src_w = image_bgr.shape[:2]
    x, y, box_w, box_h = [float(v) for v in bbox_xywh]

    scale = min((src_h - 1) / max(box_h, 1.0), (src_w - 1) / max(box_w, 1.0), float(scale))
    new_w = box_w * scale
    new_h = box_h * scale
    center_x = x + box_w / 2.0
    center_y = y + box_h / 2.0

    x1 = max(0, int(center_x - new_w / 2.0))
    y1 = max(0, int(center_y - new_h / 2.0))
    x2 = min(src_w - 1, int(center_x + new_w / 2.0))
    y2 = min(src_h - 1, int(center_y + new_h / 2.0))

    cropped = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if cropped.size == 0:
        return None
    return cropped


def _run_minifas_probs(image_bgr, bbox_xywh, model_path, scale):
    session = _load_antispoof_session(str(model_path))
    input_cfg = session.get_inputs()[0]
    # ONNX shape is typically [N, C, H, W]
    height = int(input_cfg.shape[2]) if isinstance(input_cfg.shape[2], int) else 80
    width = int(input_cfg.shape[3]) if isinstance(input_cfg.shape[3], int) else 80

    crop = _crop_face_for_antispoof(image_bgr, bbox_xywh, scale)
    if crop is None:
        return None
    # Yakhyo MiniFASNet ONNX expects raw BGR float32 (no /255)
    face = cv2.resize(crop, (width, height)).astype(np.float32)
    tensor = face.transpose(2, 0, 1)[np.newaxis, ...]
    outputs = session.run(None, {input_cfg.name: tensor})
    return _softmax(outputs[0])[0]


def _minifas_liveness(image_bgr, face_row):
    bbox_xywh = [float(v) for v in face_row[:4]]
    summed = None
    used = []
    for path, scale in _ANTISPOOF_MODELS:
        if not path.exists():
            continue
        probs = _run_minifas_probs(image_bgr, bbox_xywh, path, scale)
        if probs is None:
            continue
        summed = probs if summed is None else (summed + probs)
        used.append(path.stem)

    if summed is None or not used:
        return None, "antispoof model inference failed"

    avg = summed / float(len(used))
    label = int(np.argmax(avg))
    live_score = float(avg[1]) if len(avg) > 1 else float(avg[0])
    spoof_score = float(avg[0] + (avg[2] if len(avg) > 2 else 0.0))
    return {
        "live_score": _clamp(live_score),
        "spoof_score": _clamp(spoof_score),
        "label": "live" if label == 1 else "spoof",
        "models": used,
    }, ""


def _face_motion_patch(image_bgr, size=64):
    face = _detect_best_face(image_bgr)
    if face is None:
        return None
    x, y, w, h = [float(v) for v in face[:4]]
    img_h, img_w = image_bgr.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return {
        "patch": gray.astype(np.float32),
        "center": (x + w / 2.0, y + h / 2.0),
        "score": float(face[14]),
    }


def _motion_liveness(frames_bgr, min_motion=1.8, max_motion=40.0):
    """Passive liveness from several camera frames (face + inter-frame motion)."""
    patches = []
    for image in frames_bgr:
        info = _face_motion_patch(image)
        if info is None:
            return {
                "ok": False,
                "error": "face not detected in all frames — stay centered on camera",
            }
        patches.append(info)

    diffs = []
    center_shifts = []
    for i in range(1, len(patches)):
        mad = float(np.mean(np.abs(patches[i]["patch"] - patches[i - 1]["patch"])))
        diffs.append(mad)
        c0 = patches[i - 1]["center"]
        c1 = patches[i]["center"]
        center_shifts.append(float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])))

    motion = float(np.mean(diffs)) if diffs else 0.0
    shift = float(np.mean(center_shifts)) if center_shifts else 0.0

    # Identical stills → motion near 0. Live subject → small natural motion.
    motion_ok = min_motion <= motion <= max_motion
    # Allow tiny head/body movement; reject huge jumps between unrelated images
    shift_ok = shift <= 80.0
    live_score = _clamp(motion / 8.0) if motion_ok and shift_ok else _clamp(motion / 20.0)
    is_live = bool(motion_ok and shift_ok)

    return {
        "ok": True,
        "is_live": is_live,
        "live_score": round(float(live_score), 4),
        "spoof_score": round(float(1.0 - live_score), 4),
        "label": "live" if is_live else "spoof",
        "motion_score": round(motion, 4),
        "center_shift": round(shift, 4),
        "frame_count": len(frames_bgr),
        "engine": "motion",
    }


def run_face_detect(image_base64, min_size=60):
    try:
        image = _decode_base64_to_bgr(image_base64)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    h, w = image.shape[:2]
    if h < int(min_size) or w < int(min_size):
        return {"ok": False, "error": "image size is too small"}

    if _has_sface_pipeline():
        faces, err = _detect_faces(image)
        if faces is None:
            return {"ok": False, "error": err}
        return {
            "ok": True,
            "face_count": len(faces),
            "faces": faces,
            "engine": "yunet",
        }

    # Fallback: no detector model — report single full-frame region.
    return {
        "ok": True,
        "face_count": 1,
        "faces": [
            {
                "bbox": {"x": 0.0, "y": 0.0, "width": float(w), "height": float(h)},
                "score": 0.0,
                "landmarks": [],
            }
        ],
        "engine": "fallback",
    }


def run_liveness_check(
    image_base64=None,
    frames_base64=None,
    threshold=0.35,
    min_size=60,
    min_motion=1.8,
    max_motion=40.0,
):
    """
    Reliable path: send 3+ camera frames (~250–400ms apart) via frames_base64.
    Still photos / same frame repeated will fail the motion check.
    """
    frames_in = list(frames_base64 or [])
    if image_base64 and not frames_in:
        frames_in = [image_base64]

    if len(frames_in) < 3:
        return {
            "ok": False,
            "error": "need at least 3 frames in frames_base64 for liveness",
            "hint": "Open /test/liveness/ or capture several live camera frames ~300ms apart",
        }

    frames_bgr = []
    for item in frames_in:
        try:
            image = _decode_base64_to_bgr(item)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        h, w = image.shape[:2]
        if h < int(min_size) or w < int(min_size):
            return {"ok": False, "error": "image size is too small"}
        frames_bgr.append(image)

    if not _has_sface_pipeline():
        return {"ok": False, "error": "YuNet face detector not available"}

    result = _motion_liveness(
        frames_bgr,
        min_motion=float(min_motion),
        max_motion=float(max_motion),
    )
    if not result.get("ok"):
        return result

    # Optional MiniFAS soft signal on the last frame (print/replay cue).
    # Not used as the sole decision — motion gates is_live.
    antispoof = None
    if _has_antispoof_pipeline():
        face_row = _detect_best_face(frames_bgr[-1])
        if face_row is not None:
            antispoof, _ = _minifas_liveness(frames_bgr[-1], face_row)

    live_score = float(result["live_score"])
    is_live = bool(result["is_live"] and live_score >= float(threshold))
    payload = {
        "ok": True,
        "is_live": is_live,
        "live_score": round(live_score, 4),
        "spoof_score": round(float(result["spoof_score"]), 4),
        "label": "live" if is_live else "spoof",
        "threshold": float(threshold),
        "motion_score": result["motion_score"],
        "center_shift": result["center_shift"],
        "frame_count": result["frame_count"],
        "engine": "motion",
    }
    if antispoof:
        payload["antispoof"] = {
            "label": antispoof["label"],
            "live_score": round(float(antispoof["live_score"]), 4),
            "models": antispoof["models"],
        }
    return payload


def _vectorize_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    vector = resized.astype(np.float32).reshape(-1)
    vector -= float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def _fallback_similarity(face_a, face_b):
    vec_a = _vectorize_face(face_a)
    vec_b = _vectorize_face(face_b)
    cosine = float(np.dot(vec_a, vec_b))
    cosine = _clamp((cosine + 1.0) / 2.0)

    orb = cv2.ORB_create(nfeatures=500)
    kpa, desa = orb.detectAndCompute(face_a, None)
    kpb, desb = orb.detectAndCompute(face_b, None)
    if desa is None or desb is None or len(kpa) == 0 or len(kpb) == 0:
        return cosine
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desa, desb)
    if not matches:
        return cosine
    matches = sorted(matches, key=lambda x: x.distance)
    top = matches[: min(60, len(matches))]
    avg_dist = sum(m.distance for m in top) / float(len(top))
    orb_similarity = 1.0 - _clamp(avg_dist / 128.0)
    return _clamp(0.65 * cosine + 0.35 * orb_similarity)


def run_face_compare(image1_base64, image2_base64, threshold=0.35, min_size=60):
    try:
        image1 = _decode_base64_to_bgr(image1_base64)
        image2 = _decode_base64_to_bgr(image2_base64)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    if min(image1.shape[:2]) < int(min_size) or min(image2.shape[:2]) < int(min_size):
        return {"ok": False, "error": "image size is too small"}

    if _has_sface_pipeline():
        face1, err1 = _extract_aligned_face(image1)
        if face1 is None:
            return {"ok": False, "error": err1}
        face2, err2 = _extract_aligned_face(image2)
        if face2 is None:
            return {"ok": False, "error": err2}
        recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL_PATH), "")
        feat1 = recognizer.feature(face1)
        feat2 = recognizer.feature(face2)
        score = float(recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))
        similarity = _clamp(score)
        effective_threshold = max(0.30, float(threshold))
        engine = "sface"
    else:
        similarity = _fallback_similarity(image1, image2)
        effective_threshold = 0.78
        engine = "fallback"

    return {
        "ok": True,
        "is_match": bool(similarity >= effective_threshold),
        "similarity": round(float(similarity), 4),
        "threshold": round(float(effective_threshold), 4),
        "engine": engine,
    }
