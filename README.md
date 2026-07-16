# FaceID API

Minimal Django API for face detect, hardened liveness, and compare.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

FFmpeg must also be installed and available on `PATH`:

```bash
brew install ffmpeg
```

ONNX models live in `api/ml_models/` (YuNet, SFace, MiniFASNet). See `api/ml_models/README.md`.
The first `say_digits` request downloads the `tiny.en` faster-whisper model.

## API authentication

All `/api/v1/` endpoints require:

```http
X-Client-Code: CD82
Authorization: Bearer fid_live_xxxxxxxx
```

Create clients in Django admin (`/admin/` → API clients). The token is shown **once** on create or regenerate.

Optional `allowed_endpoints` on each client:

- empty → all endpoints
- `["detect", "liveness", "compare", "health"]` → restricted list

## Endpoints

### `GET /api/v1/health/`

### `POST /api/v1/detect/`

```json
{ "image_base64": "<base64>" }
```

Or multipart file field `image`.

### Liveness (hardened)

**1. Create challenge**

`POST /api/v1/liveness/challenge/`

```json
{
  "ok": true,
  "challenge_id": "...",
  "challenge_token": "...",
  "action": "turn_left",
  "instruction": "Slowly turn your head to YOUR left...",
  "expires_in": 90,
  "min_frames": 5,
  "required_source": "camera"
}
```

Actions: `blink` | `turn_left` | `turn_right` | `nod` | `smile` | `say_digits`

**2. Submit live camera frames while doing the action**

`POST /api/v1/liveness/`

```json
{
  "challenge_id": "...",
  "challenge_token": "...",
  "source": "camera",
  "frames_base64": ["f1", "f2", "f3", "f4", "f5", "..."]
}
```

Checks combined:
1. `source` must be `camera` (rejects file/video upload claims)
2. Challenge token valid, unused, not expired
3. Face motion across frames
4. Requested action verified from landmarks
5. MiniFASNet anti-spoof (print/replay texture)
6. Light screen-banding risk heuristic

Browser helper: `/test/liveness/`

Optional `digits_match_rate` when creating a challenge (for `say_digits` only):

- `100` → all 4 digits must match (default)
- `75` → at least 3 of 4
- `50` → at least 2 of 4

```json
POST /api/v1/liveness/challenge/
{ "digits_match_rate": 75 }
```

For `say_digits`, record camera and microphone together and submit the WebM/MP4:

`POST /api/v1/liveness/speech/` as `multipart/form-data`

- `recording`: browser `MediaRecorder` video file with audio
- `challenge_id`: challenge ID
- `challenge_token`: signed challenge token
- `source`: `camera`

The response verifies the recognized digit sequence, audio/lip synchronization,
face consistency throughout the clip, natural motion, MiniFAS anti-spoofing,
and screen-replay risk.

### `POST /api/v1/compare/`

```json
{
  "image1_base64": "<base64>",
  "image2_base64": "<base64>",
  "threshold": 0.35
}
```
