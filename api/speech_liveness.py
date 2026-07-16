"""Audio/video liveness for the random-digits challenge."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from .face_detection import (
    SFACE_MODEL_PATH,
    _detect_best_face,
    _has_antispoof_pipeline,
    _has_sface_pipeline,
    _minifas_liveness,
    _motion_liveness,
)
from .liveness import (
    ALLOWED_DIGITS_MATCH_RATES,
    _load_challenge,
    _mark_challenge_used,
    _normalize_digits_match_rate,
    _screen_replay_risk,
)
from .risk import build_risk_response

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None


MAX_RECORDING_SECONDS = 15.0
TARGET_VIDEO_FPS = 8.0
MIN_VIDEO_FRAMES = 16

_DIGIT_WORDS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "won": 1,
    "two": 2,
    "to": 2,
    "too": 2,
    "three": 3,
    "four": 4,
    "for": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "ate": 8,
    "nine": 9,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@lru_cache(maxsize=1)
def _whisper_model():
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed")
    model_name = os.environ.get("FACEID_WHISPER_MODEL", "tiny.en")
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def speech_recognition_ready() -> bool:
    return WhisperModel is not None


def _run_ffmpeg_audio(video_path: Path, wav_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise ValueError("ffmpeg is required for speech liveness") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("recording audio extraction timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"recording has no readable audio: {detail or 'ffmpeg failed'}") from exc


def _extract_video_frames(video_path: Path) -> tuple[list[np.ndarray], list[float], float]:
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    next_sample = 0.0
    last_time = 0.0

    try:
        with av.open(str(video_path)) as container:
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if video_stream is None:
                raise ValueError("recording has no video stream")

            fallback_fps = float(video_stream.average_rate or 30.0)
            for index, frame in enumerate(container.decode(video=video_stream.index)):
                timestamp = float(frame.time) if frame.time is not None else index / fallback_fps
                last_time = max(last_time, timestamp)
                if timestamp + 1e-6 < next_sample:
                    continue
                frames.append(frame.to_ndarray(format="bgr24"))
                timestamps.append(timestamp)
                next_sample = timestamp + 1.0 / TARGET_VIDEO_FPS
                if timestamp >= MAX_RECORDING_SECONDS:
                    break
    except (av.error.FFmpegError, StopIteration) as exc:
        raise ValueError("recording has no readable video") from exc

    if len(frames) < MIN_VIDEO_FRAMES:
        raise ValueError(
            f"recording is too short; need at least {MIN_VIDEO_FRAMES} sampled video frames"
        )
    return frames, timestamps, last_time


def _read_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("expected 16-bit extracted audio")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size < sample_rate:
        raise ValueError("recording audio is too short")
    return samples, sample_rate


def _transcribe_digits(wav_path: Path) -> tuple[str, list[int]]:
    model = _whisper_model()
    segments, _ = model.transcribe(
        str(wav_path),
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="The speaker reads four individual digits from zero to nine.",
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    return transcript, _digits_from_text(transcript)


def _digits_from_text(text: str) -> list[int]:
    digits: list[int] = []
    for token in re.findall(r"[a-z]+|\d+", str(text).lower()):
        if token.isdigit():
            digits.extend(int(char) for char in token)
        elif token in _DIGIT_WORDS:
            digits.append(_DIGIT_WORDS[token])
    return digits


def _required_digit_matches(total: int, match_rate: int) -> int:
    return max(1, int(round(total * match_rate / 100)))


def _count_positional_digit_matches(expected: list[int], spoken: list[int]) -> int:
    return sum(
        1
        for index, digit in enumerate(expected)
        if index < len(spoken) and spoken[index] == digit
    )


def _verify_digits_match(
    expected: list[int],
    spoken: list[int],
    match_rate: int,
) -> dict[str, Any]:
    rate = _normalize_digits_match_rate(match_rate)
    required = _required_digit_matches(len(expected), rate)
    matched = _count_positional_digit_matches(expected, spoken)
    return {
        "passed": matched >= required,
        "match_rate": rate,
        "required_matches": required,
        "matched_count": matched,
        "total_digits": len(expected),
    }


def _mouth_patch(image_bgr: np.ndarray, face_row) -> np.ndarray | None:
    face_x, face_y, face_w, face_h = [float(value) for value in face_row[:4]]
    mouth_a = np.array([float(face_row[10]), float(face_row[11])], dtype=np.float32)
    mouth_b = np.array([float(face_row[12]), float(face_row[13])], dtype=np.float32)
    center = (mouth_a + mouth_b) / 2.0
    roi_w = max(float(np.linalg.norm(mouth_b - mouth_a)) * 1.8, face_w * 0.32)
    roi_h = max(face_h * 0.22, 16.0)

    image_h, image_w = image_bgr.shape[:2]
    x1 = max(0, int(center[0] - roi_w / 2.0))
    y1 = max(0, int(center[1] - roi_h / 2.0))
    x2 = min(image_w, int(center[0] + roi_w / 2.0))
    y2 = min(image_h, int(center[1] + roi_h / 2.0))
    if x2 - x1 < 8 or y2 - y1 < 6:
        return None

    patch = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(patch, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def _mouth_motion(frames: list[np.ndarray]) -> tuple[list[float], str]:
    patches: list[np.ndarray] = []
    for image in frames:
        face = _detect_best_face(image)
        if face is None:
            return [], "face not detected throughout recording"
        patch = _mouth_patch(image, face)
        if patch is None:
            return [], "mouth region could not be tracked"
        patches.append(patch)

    movement = [
        float(np.mean(np.abs(patches[index] - patches[index - 1])))
        for index in range(1, len(patches))
    ]
    return movement, ""


def _audio_rms_bins(samples: np.ndarray, bin_count: int) -> list[float]:
    bins = np.array_split(samples, bin_count)
    return [
        float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        for chunk in bins
    ]


def _verify_audio_lip_sync(
    samples: np.ndarray,
    mouth_movement: list[float],
) -> dict[str, Any]:
    audio_rms = np.array(_audio_rms_bins(samples, len(mouth_movement)), dtype=np.float32)
    mouth = np.array(mouth_movement, dtype=np.float32)

    noise_floor = float(np.percentile(audio_rms, 25))
    audio_peak = float(np.percentile(audio_rms, 90))
    audio_threshold = noise_floor + max(0.003, (audio_peak - noise_floor) * 0.25)
    mouth_threshold = max(1.0, float(np.percentile(mouth, 45)))

    audio_active = audio_rms > audio_threshold
    mouth_active = mouth > mouth_threshold
    audio_ratio = float(np.mean(audio_active))
    mouth_ratio = float(np.mean(mouth_active))
    overlap = float(np.mean(mouth_active[audio_active])) if np.any(audio_active) else 0.0

    correlation = 0.0
    if float(np.std(audio_rms)) > 1e-5 and float(np.std(mouth)) > 1e-5:
        correlation = float(np.corrcoef(audio_rms, mouth)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    passed = bool(
        audio_peak >= 0.008
        and audio_ratio >= 0.12
        and mouth_ratio >= 0.12
        and (overlap >= 0.30 or correlation >= 0.08)
    )
    score = _clamp01(
        0.35 * min(1.0, audio_peak / 0.04)
        + 0.35 * overlap
        + 0.30 * _clamp01((correlation + 0.2) / 0.8)
    )
    return {
        "passed": passed,
        "score": round(score, 4),
        "audio_peak": round(audio_peak, 5),
        "audio_activity_ratio": round(audio_ratio, 4),
        "mouth_activity_ratio": round(mouth_ratio, 4),
        "overlap": round(overlap, 4),
        "correlation": round(correlation, 4),
    }


def _face_consistency(frames: list[np.ndarray]) -> dict[str, Any]:
    if not _has_sface_pipeline():
        return {"passed": False, "error": "SFace model not available"}

    recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL_PATH), "")
    selected_indexes = sorted({0, len(frames) // 3, (2 * len(frames)) // 3, len(frames) - 1})
    features = []
    for index in selected_indexes:
        face = _detect_best_face(frames[index])
        if face is None:
            return {"passed": False, "error": "face missing in consistency sample"}
        aligned = recognizer.alignCrop(frames[index], face)
        if aligned is None or aligned.size == 0:
            return {"passed": False, "error": "face alignment failed"}
        features.append(recognizer.feature(aligned))

    similarities = [
        float(recognizer.match(features[0], feature, cv2.FaceRecognizerSF_FR_COSINE))
        for feature in features[1:]
    ]
    minimum = min(similarities) if similarities else 0.0
    return {
        "passed": minimum >= 0.30,
        "minimum_similarity": round(minimum, 4),
        "similarities": [round(value, 4) for value in similarities],
    }


def run_speech_liveness(
    *,
    recording_bytes: bytes,
    filename: str,
    challenge_id: str,
    challenge_token: str,
    source: str,
    allow_risk: float | None = None,
    review_risk: float | None = None,
) -> dict[str, Any]:
    if str(source or "").strip().lower() != "camera":
        return {
            "ok": False,
            "is_live": False,
            "error": "source must be 'camera'",
            "checks": {"source": False},
        }

    challenge, error = _load_challenge(challenge_id, challenge_token)
    if challenge is None:
        return {"ok": False, "is_live": False, "error": error}
    if challenge.get("action") != "say_digits" or not challenge.get("digits"):
        return {
            "ok": False,
            "is_live": False,
            "error": "challenge action is not say_digits",
        }

    suffix = Path(filename or "recording.webm").suffix or ".webm"
    with tempfile.TemporaryDirectory(prefix="faceid-speech-") as temp_dir:
        video_path = Path(temp_dir) / f"recording{suffix}"
        wav_path = Path(temp_dir) / "audio.wav"
        video_path.write_bytes(recording_bytes)

        try:
            frames, _, duration = _extract_video_frames(video_path)
            _run_ffmpeg_audio(video_path, wav_path)
            samples, _ = _read_wav(wav_path)
            transcript, spoken_digits = _transcribe_digits(wav_path)
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "is_live": False, "error": str(exc)}
        except Exception as exc:
            return {
                "ok": False,
                "is_live": False,
                "error": f"speech verification failed: {exc}",
            }

    expected_digits = [int(value) for value in challenge["digits"]]
    digits_result = _verify_digits_match(
        expected_digits,
        spoken_digits,
        challenge.get("digits_match_rate", 100),
    )
    digits_match = bool(digits_result["passed"])

    mouth_movement, mouth_error = _mouth_motion(frames)
    if mouth_error:
        return {"ok": False, "is_live": False, "error": mouth_error}
    lip_sync = _verify_audio_lip_sync(samples, mouth_movement)
    consistency = _face_consistency(frames)
    motion = _motion_liveness(frames, min_motion=1.0, max_motion=45.0)

    antispoof_ok = True
    antispoof_detail = None
    if _has_antispoof_pipeline():
        middle = frames[len(frames) // 2]
        face = _detect_best_face(middle)
        if face is None:
            antispoof_ok = False
        else:
            antispoof_detail, _ = _minifas_liveness(middle, face)
            antispoof_ok = bool(
                antispoof_detail
                and not (
                    antispoof_detail["label"] == "spoof"
                    and float(antispoof_detail["spoof_score"]) >= 0.70
                )
            )

    screen_risk, _ = _screen_replay_risk(frames)
    screen_ok = screen_risk <= 0.85

    checks = {
        "source": True,
        "challenge_valid": True,
        "digits_match": digits_match,
        "lip_sync": bool(lip_sync["passed"]),
        "face_consistent": bool(consistency.get("passed")),
        "motion": bool(motion.get("ok") and motion.get("is_live")),
        "antispoof": antispoof_ok,
        "screen_ok": screen_ok,
    }

    digits_signal = _clamp01(
        digits_result["matched_count"] / max(1, digits_result["required_matches"])
    )
    antispoof_live = (
        float(antispoof_detail["live_score"]) if antispoof_detail else None
    )

    risk = build_risk_response(
        signals={
            "digits": digits_signal,
            "lip_sync": float(lip_sync["score"]),
            "face_consistency": _clamp01(float(consistency.get("minimum_similarity", 0.0))),
            "motion": float(motion.get("live_score", 0.0)) if motion.get("ok") else 0.0,
            "antispoof": antispoof_live,
            "screen": 1.0 - screen_risk,
        },
        weights={
            "digits": 1.5,
            "lip_sync": 1.3,
            "face_consistency": 1.1,
            "motion": 1.0,
            "antispoof": 1.0,
            "screen": 0.8,
        },
        allow_risk=allow_risk,
        review_risk=review_risk,
        hard_fail=not all(checks.values()),
    )
    is_live = bool(risk["is_live"])

    reasons = []
    if not digits_match:
        reasons.append(
            "spoken digits did not meet required match rate "
            f"({digits_result['matched_count']}/{digits_result['required_matches']} required at {digits_result['match_rate']}%)"
        )
    if not checks["lip_sync"]:
        reasons.append("speech and mouth movement were not synchronized")
    if not checks["face_consistent"]:
        reasons.append("the same face was not present throughout")
    if not checks["motion"]:
        reasons.append("insufficient natural face motion")
    if not antispoof_ok:
        reasons.append("antispoof model flagged likely print/replay")
    if not screen_ok:
        reasons.append("screen-replay risk too high")
    if risk["decision"] == "review":
        reasons.append("borderline liveness — manual review or retry recommended")

    if is_live:
        _mark_challenge_used(challenge["id"])

    return {
        "ok": True,
        "is_live": is_live,
        "live_score": risk["live_score"],
        "spoof_score": risk["spoof_score"],
        "risk_score": risk["risk_score"],
        "confidence": risk["confidence"],
        "decision": risk["decision"],
        "label": risk["label"],
        "action": "say_digits",
        "expected_digits": expected_digits,
        "spoken_digits": spoken_digits,
        "digits_match_rate": digits_result["match_rate"],
        "digits_required_matches": digits_result["required_matches"],
        "digits_matched_count": digits_result["matched_count"],
        "allowed_match_rates": list(ALLOWED_DIGITS_MATCH_RATES),
        "transcript": transcript,
        "duration": round(float(duration), 2),
        "engine": "speech+lip-sync+face+antispoof",
        "checks": checks,
        "risk": risk,
        "lip_sync": lip_sync,
        "face_consistency": consistency,
        "antispoof": antispoof_detail,
        "screen_risk": round(screen_risk, 4),
        "reason": None if is_live else "; ".join(reasons),
    }
