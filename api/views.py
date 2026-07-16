import base64

from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from .face_detection import run_face_compare, run_face_detect
from .liveness import create_liveness_challenge, run_hardened_liveness
from .speech_liveness import run_speech_liveness, speech_recognition_ready


def _file_to_base64(uploaded):
    return base64.b64encode(uploaded.read()).decode("ascii")


def _require_image(request, key="image_base64", file_key="image"):
    """Accept JSON/form base64 (`image_base64`) or multipart file (`image`)."""
    uploaded = request.FILES.get(file_key)
    if uploaded is not None:
        return _file_to_base64(uploaded), None

    value = request.data.get(key)
    if value:
        return value, None

    return None, Response(
        {
            "ok": False,
            "error": f"{key} or file field '{file_key}' is required",
        },
        status=status.HTTP_400_BAD_REQUEST,
    )


@api_view(["POST"])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def detect_face(request):
    image, error = _require_image(request)
    if error:
        return error

    result = run_face_detect(
        image,
        min_size=request.data.get("min_size", 60),
    )
    code = status.HTTP_200_OK if result.get("ok") else status.HTTP_400_BAD_REQUEST
    return Response(result, status=code)


@api_view(["POST"])
@parser_classes([JSONParser, FormParser])
def create_challenge(request):
    return Response(
        create_liveness_challenge(
            digits_match_rate=request.data.get("digits_match_rate", 100),
        )
    )


@api_view(["POST"])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def detect_liveness(request):
    frames = request.data.get("frames_base64")
    if isinstance(frames, str):
        frames = [frames]

    if not frames:
        return Response(
            {
                "ok": False,
                "is_live": False,
                "error": "frames_base64 is required (5+ live camera frames)",
                "hint": "1) POST /api/v1/liveness/challenge/  2) perform action  3) submit frames with challenge_token and source=camera",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    result = run_hardened_liveness(
        frames_base64=frames,
        challenge_id=str(request.data.get("challenge_id") or ""),
        challenge_token=str(request.data.get("challenge_token") or ""),
        source=str(request.data.get("source") or ""),
        min_size=int(request.data.get("min_size", 60)),
        min_motion=float(request.data.get("min_motion", 1.5)),
        max_motion=float(request.data.get("max_motion", 45.0)),
        antispoof_min_live=float(request.data.get("antispoof_min_live", 0.70)),
        max_screen_risk=float(request.data.get("max_screen_risk", 0.85)),
        allow_risk=request.data.get("allow_risk"),
        review_risk=request.data.get("review_risk"),
    )
    code = status.HTTP_200_OK if result.get("ok") else status.HTTP_400_BAD_REQUEST
    return Response(result, status=code)


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def detect_speech_liveness(request):
    recording = request.FILES.get("recording")
    if recording is None:
        return Response(
            {
                "ok": False,
                "is_live": False,
                "error": "multipart file field 'recording' is required",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )
    if recording.size > 30 * 1024 * 1024:
        return Response(
            {"ok": False, "is_live": False, "error": "recording exceeds 30 MB"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    result = run_speech_liveness(
        recording_bytes=recording.read(),
        filename=recording.name,
        challenge_id=str(request.data.get("challenge_id") or ""),
        challenge_token=str(request.data.get("challenge_token") or ""),
        source=str(request.data.get("source") or ""),
        allow_risk=request.data.get("allow_risk"),
        review_risk=request.data.get("review_risk"),
    )
    code = status.HTTP_200_OK if result.get("ok") else status.HTTP_400_BAD_REQUEST
    return Response(result, status=code)


@api_view(["POST"])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def compare_faces(request):
    image1, error1 = _require_image(request, "image1_base64", "image1")
    if error1:
        return error1
    image2, error2 = _require_image(request, "image2_base64", "image2")
    if error2:
        return error2

    result = run_face_compare(
        image1,
        image2,
        threshold=request.data.get("threshold", 0.35),
        min_size=request.data.get("min_size", 60),
    )
    code = status.HTTP_200_OK if result.get("ok") else status.HTTP_400_BAD_REQUEST
    return Response(result, status=code)


@api_view(["GET"])
def health(request):
    from .face_detection import _has_antispoof_pipeline, _has_sface_pipeline

    return Response(
        {
            "ok": True,
            "service": "FaceID",
            "models_ready": _has_sface_pipeline(),
            "liveness_ready": _has_sface_pipeline(),
            "antispoof_ready": _has_antispoof_pipeline(),
            "speech_liveness_ready": speech_recognition_ready(),
            "liveness_mode": "challenge+motion+antispoof",
        }
    )


def liveness_test_page(request):
    return render(request, "api/liveness_test.html")
