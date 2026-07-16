"""Hardened liveness: challenge-response + motion + MiniFAS + camera source gate."""

from __future__ import annotations

import json
import secrets
import uuid
from typing import Any

import cv2
import numpy as np
from django.core import signing
from django.core.cache import cache

from .face_detection import (
    _decode_base64_to_bgr,
    _detect_best_face,
    _has_antispoof_pipeline,
    _has_sface_pipeline,
    _minifas_liveness,
    _motion_liveness,
)
from .risk import build_risk_response

CHALLENGE_SALT = "faceid.liveness.challenge"
CHALLENGE_MAX_AGE = 180  # seconds
CHALLENGE_CACHE_PREFIX = "liveness:challenge:"
MIN_FRAMES = 5

ACTIONS = ("blink", "turn_left", "turn_right", "nod", "smile", "say_digits")

ACTION_INSTRUCTIONS = {
    "blink": "Blink your eyes clearly a couple of times.",
    "turn_left": "Slowly turn your head to YOUR left, then face the camera again.",
    "turn_right": "Slowly turn your head to YOUR right, then face the camera again.",
    "nod": "Slowly nod your head up and down.",
    "smile": "Start with a neutral face, then smile clearly and hold it briefly.",
    "say_digits": "Say the displayed digits clearly while looking at the camera.",
}

ALLOWED_DIGITS_MATCH_RATES = (100, 75, 50)


def _normalize_digits_match_rate(rate) -> int:
    try:
        value = int(rate)
    except (TypeError, ValueError):
        return 100
    return value if value in ALLOWED_DIGITS_MATCH_RATES else 100


def _signer():
    return signing.TimestampSigner(salt=CHALLENGE_SALT)


def create_liveness_challenge(digits_match_rate: int = 100) -> dict[str, Any]:
    action = secrets.choice(ACTIONS)
    challenge_id = str(uuid.uuid4())
    digits = [secrets.randbelow(10) for _ in range(4)] if action == "say_digits" else None
    match_rate = _normalize_digits_match_rate(digits_match_rate) if action == "say_digits" else None
    payload = {
        "id": challenge_id,
        "action": action,
        "nonce": secrets.token_hex(8),
        "digits": digits,
        "digits_match_rate": match_rate,
    }
    token = _signer().sign(json.dumps(payload, separators=(",", ":")))
    cache.set(
        f"{CHALLENGE_CACHE_PREFIX}{challenge_id}",
        {
            "action": action,
            "digits": digits,
            "digits_match_rate": match_rate,
            "used": False,
            "nonce": payload["nonce"],
        },
        timeout=CHALLENGE_MAX_AGE,
    )
    response = {
        "ok": True,
        "challenge_id": challenge_id,
        "challenge_token": token,
        "action": action,
        "instruction": ACTION_INSTRUCTIONS[action],
        "expires_in": CHALLENGE_MAX_AGE,
        "min_frames": MIN_FRAMES,
        "required_source": "camera",
    }
    if digits is not None:
        required = max(1, int(round(len(digits) * match_rate / 100)))
        response["digits"] = digits
        response["digits_text"] = " ".join(str(digit) for digit in digits)
        response["digits_match_rate"] = match_rate
        response["digits_required_matches"] = required
        response["instruction"] = (
            f"Say these digits clearly: {' '.join(str(digit) for digit in digits)} "
            f"(at least {required} of {len(digits)} must match)"
        )
        response["requires_audio"] = True
    return response


def _load_challenge(challenge_id: str, challenge_token: str) -> tuple[dict | None, str]:
    if not challenge_id or not challenge_token:
        return None, "challenge_id and challenge_token are required"

    try:
        raw = _signer().unsign(challenge_token, max_age=CHALLENGE_MAX_AGE)
        data = json.loads(raw)
    except signing.SignatureExpired:
        return None, "challenge expired — request a new one"
    except (signing.BadSignature, json.JSONDecodeError, TypeError):
        return None, "invalid challenge_token"

    if str(data.get("id")) != str(challenge_id):
        return None, "challenge_id does not match token"

    cached = cache.get(f"{CHALLENGE_CACHE_PREFIX}{challenge_id}")
    if not cached:
        return None, "challenge expired or unknown"
    if cached.get("used"):
        return None, "challenge already used"
    if cached.get("action") != data.get("action"):
        return None, "challenge mismatch"
    if cached.get("digits") != data.get("digits"):
        return None, "challenge digits mismatch"
    cached_rate = cached.get("digits_match_rate")
    token_rate = data.get("digits_match_rate")
    if cached_rate is not None and token_rate is not None and cached_rate != token_rate:
        return None, "challenge digits match rate mismatch"
    return {
        "id": challenge_id,
        "action": data["action"],
        "nonce": data.get("nonce"),
        "digits": data.get("digits"),
        "digits_match_rate": _normalize_digits_match_rate(
            cached_rate if cached_rate is not None else token_rate
        ),
    }, ""


def _mark_challenge_used(challenge_id: str) -> None:
    cached = cache.get(f"{CHALLENGE_CACHE_PREFIX}{challenge_id}")
    if cached is not None:
        cached["used"] = True
        cache.set(f"{CHALLENGE_CACHE_PREFIX}{challenge_id}", cached, timeout=CHALLENGE_MAX_AGE)


def _geom_from_face(face_row) -> dict[str, float] | None:
    x, y, w, h = [float(v) for v in face_row[:4]]
    if w < 20 or h < 20:
        return None
    lx, ly = float(face_row[4]), float(face_row[5])
    rx, ry = float(face_row[6]), float(face_row[7])
    nx, ny = float(face_row[8]), float(face_row[9])
    mx1, my1 = float(face_row[10]), float(face_row[11])
    mx2, my2 = float(face_row[12]), float(face_row[13])
    cx, cy = x + w / 2.0, y + h / 2.0
    return {
        "w": w,
        "h": h,
        "nose_x": (nx - cx) / w,
        "nose_y": (ny - cy) / h,
        "center_x": cx,
        "center_y": cy,
        "eye_dist": float(np.hypot(rx - lx, ry - ly) / w),
        "mouth_w": float(np.hypot(mx2 - mx1, my2 - my1) / w),
        "left_eye": (lx, ly),
        "right_eye": (rx, ry),
    }


def _eye_intensity(image_bgr, eye_xy, face_w) -> float:
    ex, ey = eye_xy
    r = max(4, int(face_w * 0.08))
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, int(ex - r)), max(0, int(ey - r))
    x2, y2 = min(w, int(ex + r)), min(h, int(ey + r))
    patch = image_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _collect_series(frames_bgr: list) -> tuple[list[dict], list[float] | None, str]:
    geoms = []
    eye_vals = []
    for image in frames_bgr:
        face = _detect_best_face(image)
        if face is None:
            return [], None, "face not detected in all frames — stay centered"
        geom = _geom_from_face(face)
        if geom is None:
            return [], None, "face too small"
        geoms.append(geom)
        left_i = _eye_intensity(image, geom["left_eye"], geom["w"])
        right_i = _eye_intensity(image, geom["right_eye"], geom["w"])
        eye_vals.append((left_i + right_i) / 2.0)
    return geoms, eye_vals, ""


def _verify_blink(eye_vals: list[float]) -> tuple[bool, float, str]:
    if len(eye_vals) < 4:
        return False, 0.0, "not enough frames for blink"
    arr = np.array(eye_vals, dtype=np.float32)
    baseline = float(np.percentile(arr, 75))
    trough = float(np.min(arr))
    drop = baseline - trough
    # Relatively darker eye region once = blink cue
    score = _clamp01(drop / max(12.0, baseline * 0.12))
    ok = drop >= max(8.0, baseline * 0.06)
    return ok, score, "blink detected" if ok else "blink not detected — blink clearly"


def _verify_turn(geoms: list[dict], direction: str) -> tuple[bool, float, str]:
    series = np.array([g["nose_x"] for g in geoms], dtype=np.float32)
    start = float(np.mean(series[:2]))
    peak = float(np.min(series)) if direction == "turn_left" else float(np.max(series))
    delta = start - peak if direction == "turn_left" else peak - start
    # Normalized nose offset change ~0.08–0.25 for a clear turn
    score = _clamp01(delta / 0.18)
    ok = delta >= 0.07
    label = "left" if direction == "turn_left" else "right"
    return ok, score, f"turn {label} detected" if ok else f"turn {label} not detected — turn more"


def _verify_nod(geoms: list[dict]) -> tuple[bool, float, str]:
    series = np.array([g["nose_y"] for g in geoms], dtype=np.float32)
    amplitude = float(np.max(series) - np.min(series))
    score = _clamp01(amplitude / 0.14)
    ok = amplitude >= 0.055
    return ok, score, "nod detected" if ok else "nod not detected — nod clearly"


def _verify_smile(geoms: list[dict]) -> tuple[bool, float, str]:
    series = np.array([g["mouth_w"] for g in geoms], dtype=np.float32)
    baseline_count = max(2, min(4, len(series) // 3))
    baseline = float(np.mean(series[:baseline_count]))
    peak = float(np.max(series[baseline_count:])) if len(series) > baseline_count else baseline
    increase = peak - baseline
    # YuNet mouth-corner distance normalized by detected face width.
    score = _clamp01(increase / 0.07)
    ok = increase >= 0.025
    return ok, score, "smile detected" if ok else "smile not detected — begin neutral, then smile wider"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _screen_replay_risk(frames_bgr: list) -> tuple[float, dict]:
    """
    Light heuristic for screen replay: unusually periodic high-frequency patterns
    and unusually stable inter-frame screen refresh banding are weak cues.
    Returns risk in [0,1] (higher = more likely screen).
    """
    risks = []
    for image in frames_bgr[:: max(1, len(frames_bgr) // 3)]:
        face = _detect_best_face(image)
        if face is None:
            continue
        x, y, w, h = [int(v) for v in face[:4]]
        crop = image[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Horizontal banding energy (common in phone/monitor capture)
        row_mean = gray.mean(axis=1)
        row_ac = row_mean - row_mean.mean()
        # Strong sinusoidal row pattern → higher risk
        fft = np.abs(np.fft.rfft(row_ac))
        if fft.size > 3:
            band = float(np.max(fft[2:]) / (np.sum(fft[1:]) + 1e-6))
            risks.append(_clamp01(band * 2.5))
    risk = float(np.mean(risks)) if risks else 0.0
    return risk, {"banding_risk": round(risk, 4)}


def verify_challenge_action(action: str, frames_bgr: list) -> dict[str, Any]:
    geoms, eye_vals, err = _collect_series(frames_bgr)
    if err:
        return {"ok": False, "passed": False, "error": err}

    if action == "blink":
        passed, score, detail = _verify_blink(eye_vals or [])
    elif action == "turn_left":
        passed, score, detail = _verify_turn(geoms, "turn_left")
    elif action == "turn_right":
        passed, score, detail = _verify_turn(geoms, "turn_right")
    elif action == "nod":
        passed, score, detail = _verify_nod(geoms)
    elif action == "smile":
        passed, score, detail = _verify_smile(geoms)
    else:
        return {"ok": False, "passed": False, "error": f"unknown action: {action}"}

    return {
        "ok": True,
        "passed": bool(passed),
        "action": action,
        "score": round(float(score), 4),
        "detail": detail,
    }


def run_hardened_liveness(
    *,
    frames_base64: list[str],
    challenge_id: str,
    challenge_token: str,
    source: str = "",
    min_size: int = 60,
    min_motion: float = 1.5,
    max_motion: float = 45.0,
    antispoof_min_live: float = 0.35,
    max_screen_risk: float = 0.85,
    allow_risk: float | None = None,
    review_risk: float | None = None,
) -> dict[str, Any]:
    source_norm = str(source or "").strip().lower()
    if source_norm != "camera":
        return {
            "ok": False,
            "is_live": False,
            "error": "source must be 'camera' (file/video uploads are rejected)",
            "checks": {"source": False},
        }

    challenge, err = _load_challenge(challenge_id, challenge_token)
    if challenge is None:
        return {"ok": False, "is_live": False, "error": err, "checks": {"challenge": False}}

    frames_in = list(frames_base64 or [])
    if len(frames_in) < MIN_FRAMES:
        return {
            "ok": False,
            "is_live": False,
            "error": f"need at least {MIN_FRAMES} frames_base64 while performing the challenge",
            "checks": {"frames": False},
        }

    if not _has_sface_pipeline():
        return {"ok": False, "is_live": False, "error": "YuNet face detector not available"}

    frames_bgr = []
    for item in frames_in:
        try:
            image = _decode_base64_to_bgr(item)
        except ValueError as exc:
            return {"ok": False, "is_live": False, "error": str(exc)}
        h, w = image.shape[:2]
        if h < int(min_size) or w < int(min_size):
            return {"ok": False, "is_live": False, "error": "image size is too small"}
        frames_bgr.append(image)

    checks: dict[str, Any] = {
        "source": True,
        "challenge_valid": True,
        "motion": False,
        "action": False,
        "antispoof": None,
        "screen_risk": None,
    }

    motion = _motion_liveness(frames_bgr, min_motion=min_motion, max_motion=max_motion)
    if not motion.get("ok"):
        return {
            "ok": False,
            "is_live": False,
            "error": motion.get("error", "motion check failed"),
            "action": challenge["action"],
            "checks": checks,
        }
    checks["motion"] = bool(motion.get("is_live"))
    checks["motion_score"] = motion.get("motion_score")

    action_result = verify_challenge_action(challenge["action"], frames_bgr)
    if not action_result.get("ok"):
        return {
            "ok": False,
            "is_live": False,
            "error": action_result.get("error"),
            "action": challenge["action"],
            "checks": checks,
        }
    checks["action"] = bool(action_result.get("passed"))
    checks["action_score"] = action_result.get("score")
    checks["action_detail"] = action_result.get("detail")

    antispoof_ok = True
    antispoof_payload = None
    if _has_antispoof_pipeline():
        face_row = _detect_best_face(frames_bgr[len(frames_bgr) // 2])
        if face_row is None:
            antispoof_ok = False
        else:
            antispoof_payload, _ = _minifas_liveness(frames_bgr[len(frames_bgr) // 2], face_row)
            if antispoof_payload is None:
                antispoof_ok = False
            else:
                spoof_score = float(antispoof_payload["spoof_score"])
                # Fail only on confident texture spoof (print / replay cues)
                antispoof_ok = not (
                    antispoof_payload["label"] == "spoof" and spoof_score >= float(antispoof_min_live)
                )
        checks["antispoof"] = antispoof_ok
        if antispoof_payload:
            checks["antispoof_detail"] = {
                "label": antispoof_payload["label"],
                "live_score": round(float(antispoof_payload["live_score"]), 4),
                "spoof_score": round(float(antispoof_payload["spoof_score"]), 4),
            }

    screen_risk, screen_meta = _screen_replay_risk(frames_bgr)
    screen_ok = screen_risk <= float(max_screen_risk)
    checks["screen_risk"] = round(screen_risk, 4)
    checks["screen_ok"] = screen_ok
    checks.update(screen_meta)

    antispoof_live = None
    if antispoof_payload:
        antispoof_live = float(antispoof_payload.get("live_score") or 0.0)

    risk = build_risk_response(
        signals={
            "motion": float(motion.get("live_score") or 0.0),
            "action": float(action_result.get("score") or 0.0),
            "antispoof": antispoof_live,
            "screen": 1.0 - screen_risk,
        },
        weights={
            "motion": 1.2,
            "action": 1.4,
            "antispoof": 1.0,
            "screen": 0.8,
        },
        allow_risk=allow_risk,
        review_risk=review_risk,
        hard_fail=not (checks["motion"] and checks["action"] and antispoof_ok and screen_ok),
    )
    is_live = bool(risk["is_live"])

    if is_live:
        _mark_challenge_used(challenge["id"])

    reasons = []
    if not checks["motion"]:
        reasons.append("insufficient face motion between frames")
    if not checks["action"]:
        reasons.append(action_result.get("detail") or "challenge action not verified")
    if checks["antispoof"] is False:
        reasons.append("antispoof model flagged likely print/replay")
    if not screen_ok:
        reasons.append("screen-replay risk too high")
    if risk["decision"] == "review":
        reasons.append("borderline liveness — manual review or retry recommended")

    return {
        "ok": True,
        "is_live": is_live,
        "live_score": risk["live_score"],
        "spoof_score": risk["spoof_score"],
        "risk_score": risk["risk_score"],
        "confidence": risk["confidence"],
        "decision": risk["decision"],
        "label": risk["label"],
        "action": challenge["action"],
        "engine": "challenge+motion+antispoof",
        "frame_count": len(frames_bgr),
        "checks": checks,
        "risk": risk,
        "reason": None if is_live else "; ".join(reasons) if reasons else None,
    }
