from dataclasses import dataclass
from email.mime.text import MIMEText

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Cookie,
    Response,
    Depends,
    UploadFile,
    File,
    Form,
    Header,
    Query,
    Security,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Request
from datetime import datetime, timedelta
import json
import asyncio
import uuid
import socket
import ipaddress
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional
import secrets
import cv2
import numpy as np
import base64
import pickle
import os
import time
import hmac
import hashlib
from pathlib import Path
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
WORKSPACE_ROOT = PROJECT_ROOT.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    # Optional dependency; environment variables can still be set by the shell.
    pass

from app.database import (
    get_user_by_username, save_conversation, get_recent_context,
    create_google_user, get_user_by_google_id, get_user_google_tokens, update_google_tokens,
    get_all_users, delete_user_by_id, admin_update_user, update_user_preferences,
    get_user_news_preferences, update_user_news_preferences,
)
from app.services.google_oauth import build_auth_url_with_verifier, exchange_code_for_tokens
from langchain_core.messages import HumanMessage, AIMessage

# ── Server-side STT via faster_whisper (shared with CLI) ──
_whisper_model = None

def _get_whisper_model():
    """Lazy-load faster_whisper model (base, CPU, int8)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("[STT] Loading Whisper model (base, CPU, int8)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("[STT] ✅ Whisper model loaded.")
    return _whisper_model

def transcribe_audio_bytes(audio_bytes: bytes) -> str | None:
    """Transcribe raw audio bytes (WAV or webm) using faster_whisper. Returns text or None."""
    import tempfile, wave, io

    try:
        model = _get_whisper_model()

        # Browser MediaRecorder sends webm/opus — write to temp file for ffmpeg decode
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        segments, info = model.transcribe(tmp_path, beam_size=5, language="en")
        seg_list = list(segments)
        text = " ".join(seg.text.strip() for seg in seg_list).strip()

        os.unlink(tmp_path)
        return text if text else None
    except Exception as e:
        print(f"[STT] Error: {e}")
        return None

# Import calendar functions
try:
    from app.calendar_service import (
        get_todays_events,
        get_upcoming_events,
        get_events_in_range,
        get_calendar_event,
        create_calendar_event,
        update_calendar_event,
        delete_calendar_event,
        set_current_user as set_calendar_current_user,
        GoogleReauthRequiredError,
    )
    CALENDAR_AVAILABLE = True
    print("[DEBUG] ✅ Calendar integration loaded successfully")
except Exception as e:
    CALENDAR_AVAILABLE = False
    GoogleReauthRequiredError = RuntimeError
    print(f"[DEBUG] ❌ Calendar integration not available: {e}")

# Initialize face recognition (InsightFace for detection)
try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    print("[DEBUG] Initializing InsightFace...")

    def _prepare_insightface(instance):
        # Different insightface versions accept providers on different methods.
        prepare_attempts = [
            {"ctx_id": -1, "det_size": (640, 640), "providers": ["CPUExecutionProvider"]},
            {"ctx_id": -1, "det_size": (640, 640)},
            {"ctx_id": 0, "det_size": (640, 640)},
        ]
        last_error = None
        for kwargs in prepare_attempts:
            try:
                instance.prepare(**kwargs)
                return
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    try:
        face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    except TypeError as exc:
        if "providers" in str(exc):
            face_app = FaceAnalysis(name='buffalo_sc')
        else:
            raise

    _prepare_insightface(face_app)
    FACE_RECOGNITION_AVAILABLE = True
    print("[DEBUG] ✅ Face recognition ready!")
except Exception as e:
    FACE_RECOGNITION_AVAILABLE = False
    face_app = None
    face_align = None
    print(f"[DEBUG] Face recognition not available: {e}")

FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.40"))
ACTIVE_FACE_MODEL_NAME = "insightface_default"
ACTIVE_FACE_MODEL_PATH = ""


def _normalize_embedding_vector(embedding):
    vec = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return vec
    return vec / norm
    print(f"[DEBUG] ❌ Face recognition not available: {e}")

# ── Custom Face Model ────────────────────────────────────────────────────────
try:
    import torch
    from torchvision import transforms
    from PIL import Image
    from app.ml.face_model import FaceEmbeddingModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[DEBUG] Activating best available face backbone...")

    custom_face_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    def _iter_face_model_candidates():
        candidate_names = [
            "best_acc_model.pth",
            "best_face_model.pth",
            "face_embedding_backbone.pth",
            "best_nepali_face_model.pth",
            "best face ko.pth",
        ]
        search_dirs = [MODELS_DIR, PROJECT_ROOT, WORKSPACE_ROOT]
        seen = set()
        for name in candidate_names:
            for directory in search_dirs:
                candidate = directory / name
                if candidate.exists() and candidate not in seen:
                    seen.add(candidate)
                    yield candidate

    def _torch_load_checkpoint(path: Path):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    def _extract_runtime_face_state_dict(checkpoint):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
            checkpoint = checkpoint["model_state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint does not contain a state dict")

        translated = {}
        for key, value in checkpoint.items():
            new_key = None
            if key.startswith("backbone.backbone."):
                new_key = key[len("backbone."):]
            elif key.startswith("backbone.projection."):
                new_key = key[len("backbone."):]
            elif key.startswith("backbone.pool."):
                new_key = key[len("backbone."):]
            elif key.startswith("backbone.features."):
                new_key = "backbone." + key[len("backbone.features."):]
            elif key.startswith("backbone."):
                new_key = key
            elif key.startswith("projection.") or key.startswith("pool."):
                new_key = key
            if new_key:
                translated[new_key] = value
        if not translated:
            raise ValueError("checkpoint does not match FaceEmbeddingModel")
        return translated

    def _load_best_face_model():
        candidate_paths = list(_iter_face_model_candidates())
        if not candidate_paths:
            raise FileNotFoundError("no face model checkpoints were found")

        for candidate in candidate_paths:
            try:
                checkpoint = _torch_load_checkpoint(candidate)
                state_dict = _extract_runtime_face_state_dict(checkpoint)
                model = FaceEmbeddingModel(embedding_size=512).to(device)
                model_keys = set(model.state_dict().keys())
                matched_keys = model_keys.intersection(state_dict.keys())
                coverage = len(matched_keys) / max(1, len(model_keys))
                if coverage < 0.85:
                    raise ValueError(f"insufficient key coverage ({coverage:.0%})")

                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                required_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
                if required_missing:
                    raise ValueError(f"missing runtime weights: {required_missing[:5]}")
                if unexpected:
                    raise ValueError(f"unexpected weights: {unexpected[:5]}")

                model.eval()

                meta_bits = []
                if isinstance(checkpoint, dict) and checkpoint.get("val_acc") is not None:
                    meta_bits.append(f"val_acc={float(checkpoint['val_acc']):.4f}")
                if isinstance(checkpoint, dict) and checkpoint.get("val_loss") is not None:
                    meta_bits.append(f"val_loss={float(checkpoint['val_loss']):.4f}")
                meta_suffix = f" ({', '.join(meta_bits)})" if meta_bits else ""
                print(f"[DEBUG] Loaded face model: {candidate}{meta_suffix}")
                return model, candidate
            except Exception as exc:
                print(f"[DEBUG] Skipping face model {candidate}: {exc}")

        raise RuntimeError("no compatible face model checkpoints could be loaded")

    custom_face_model, selected_face_model_path = _load_best_face_model()
    ACTIVE_FACE_MODEL_NAME = selected_face_model_path.name
    ACTIVE_FACE_MODEL_PATH = str(selected_face_model_path)
    print(f"[DEBUG] Active custom face model: {ACTIVE_FACE_MODEL_NAME}")
    USE_CUSTOM_FACE_MODEL = True
    print("[DEBUG] ✅ Custom Face Model Loaded Successfully!")
except Exception as e:
    print(f"[DEBUG] ⚠️ Custom face model failed to load (will fallback to InsightFace): {e}")
    custom_face_model = None
    USE_CUSTOM_FACE_MODEL = False

def _extract_face_crop(frame, face):
    if face_align is not None and getattr(face, "kps", None) is not None:
        try:
            aligned = face_align.norm_crop(frame, landmark=face.kps, image_size=112)
            if aligned is not None and getattr(aligned, "size", 0):
                return aligned
        except Exception as exc:
            print(f"[DEBUG] Face alignment fallback: {exc}")

    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid face crop bounds")
    return frame[y1:y2, x1:x2]


def get_face_embedding(frame, face):
    """
    Given a raw BGR frame and an InsightFace detection object (`face`),
    generates a 512-D unit-norm embedding.
    Uses our custom fine-tuned PyTorch backbone if available, otherwise maps to InsightFace's default.
    """
    if USE_CUSTOM_FACE_MODEL:
        try:
            face_crop = _extract_face_crop(frame, face)
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            
            img_t = custom_face_transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = custom_face_model(img_t).cpu().numpy().squeeze()
            return _normalize_embedding_vector(emb)
        except Exception as exc:
            print(f"[DEBUG] Custom face extraction error {exc}, falling back.")
            pass
            
    # Fallback to default InsightFace L2-normalized embedding
    return _normalize_embedding_vector(face.embedding)


def _prepare_enrollment_embeddings(embeddings):
    cleaned = [_normalize_embedding_vector(emb) for emb in embeddings if emb is not None]
    if len(cleaned) <= 2:
        return cleaned

    centroid = _normalize_embedding_vector(np.mean(cleaned, axis=0))
    ranked = sorted(
        ((float(np.dot(emb, centroid)), emb) for emb in cleaned),
        key=lambda item: item[0],
        reverse=True,
    )

    keep = [emb for score, emb in ranked if score >= 0.45]
    if len(keep) < min(3, len(ranked)):
        keep = [emb for _, emb in ranked[:min(5, len(ranked))]]
    return keep[:8]


def _score_enrolled_embeddings(test_emb, embeddings):
    cleaned = [_normalize_embedding_vector(emb) for emb in embeddings if emb is not None]
    if not cleaned:
        return 0.0

    test_emb = _normalize_embedding_vector(test_emb)
    emb_matrix = np.stack(cleaned, axis=0)
    similarities = emb_matrix @ test_emb
    centroid = _normalize_embedding_vector(np.mean(emb_matrix, axis=0))
    centroid_similarity = float(np.dot(centroid, test_emb))
    top_k = similarities[np.argsort(similarities)[-min(3, len(similarities)):]]
    top_k_mean = float(np.mean(top_k))
    best_similarity = float(np.max(similarities))

    # Blend the user's centroid with the strongest captured frames.
    return (0.5 * centroid_similarity) + (0.35 * top_k_mean) + (0.15 * best_similarity)


def _find_best_face_match(test_emb):
    best_match = None
    best_similarity = 0.0

    for username, embeddings in face_users_db.items():
        if not embeddings:
            continue
        matched_user = get_user_by_username(username)
        if not matched_user or not matched_user.get("google_id"):
            continue

        similarity = _score_enrolled_embeddings(test_emb, embeddings)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = username

    return best_match, best_similarity

# Face database
FACE_DB_FILE = DATA_DIR / "face_database.pkl"

def load_face_database():
    if os.path.exists(FACE_DB_FILE):
        try:
            with open(FACE_DB_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    return {}

def save_face_database(db):
    with open(FACE_DB_FILE, 'wb') as f:
        pickle.dump(db, f)

face_users_db = load_face_database()

# Face detection cache (username -> last_seen_time)
face_detection_cache = {}

app = FastAPI()
print(f"[DEBUG] Calendar available: {CALENDAR_AVAILABLE}")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def render_template(name: str, request: Request, context: Optional[dict] = None, **kwargs):
    """Compatibility wrapper for Starlette TemplateResponse API differences."""
    merged_context = {"request": request}
    if context:
        merged_context.update(context)

    try:
        # Newer Starlette/FastAPI style.
        return templates.TemplateResponse(
            request=request,
            name=name,
            context=merged_context,
            **kwargs,
        )
    except TypeError as exc:
        # Older Starlette style.
        message = str(exc)
        if ("request" not in message) and ("multiple values" not in message):
            raise
        return templates.TemplateResponse(name, merged_context, **kwargs)

# Session storage (in production, use Redis or database)
sessions = {}
revoked_sessions = set()
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
SESSION_SIGNING_SECRET = os.getenv("SESSION_SIGNING_SECRET", os.getenv("SECRET_KEY", "change-me-session-secret"))
voice_service_warm_lock = asyncio.Lock()
voice_service_state = {
    "ready": False,
    "loading": False,
    "stt_ready": False,
    "tts_ready": False,
    "last_error": "",
}
session_auth_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    token: Optional[str]
    username: Optional[str]


class MailSendRequest(BaseModel):
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(default="", description="Email subject")
    body: str = Field(default="", description="Plain-text email body")
    topic: str = Field(default="", description="Fallback topic used when body is empty")
    additional_context: str = Field(default="", description="Optional extra context appended to the body")


class CalendarEventCreateRequest(BaseModel):
    summary: str = Field(..., min_length=1, description="Calendar event title")
    start_time: datetime = Field(..., description="ISO datetime for event start")
    end_time: datetime = Field(..., description="ISO datetime for event end")
    description: str = Field(default="", description="Optional event description")
    location: str = Field(default="", description="Optional event location")
    timezone: str = Field(default="Asia/Kathmandu", description="IANA timezone name")


class CalendarEventUpdateRequest(BaseModel):
    summary: Optional[str] = Field(default=None, description="Updated event title")
    start_time: Optional[datetime] = Field(default=None, description="Updated ISO start datetime")
    end_time: Optional[datetime] = Field(default=None, description="Updated ISO end datetime")
    description: Optional[str] = Field(default=None, description="Updated event description")
    location: Optional[str] = Field(default=None, description="Updated event location")
    timezone: str = Field(default="Asia/Kathmandu", description="IANA timezone name")


class NewsPreferenceUpdateRequest(BaseModel):
    interests: Optional[str] = Field(default=None, description="Comma-separated interests for personalized news")
    country: Optional[str] = Field(default=None, description="Country name or 2-letter code for personalized news")


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static"):
        return await call_next(request)

    started_at = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        status = getattr(response, "status_code", "ERR")
        print(f"[TIMING] HTTP {request.method} {path} -> {status} in {_elapsed_ms(started_at):.0f} ms")


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    pad = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + pad)


def _decode_session_claims(token: str) -> Optional[dict]:
    if not token or not token.startswith("s1."):
        return None

    try:
        _, payload_b64, sig_b64 = token.split(".", 2)
        expected_sig = hmac.new(
            SESSION_SIGNING_SECRET.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        provided_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(provided_sig, expected_sig):
            return None

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if not isinstance(payload, dict):
            return None

        username = str(payload.get("u") or "").strip()
        issued_at = int(payload.get("iat") or 0)
        if not username or issued_at <= 0:
            return None

        if SESSION_TTL_SECONDS > 0 and (issued_at + SESSION_TTL_SECONDS) < int(time.time()):
            return None

        return {"u": username, "iat": issued_at}
    except Exception:
        return None


def _issue_session_token(username: str) -> str:
    # Tokens are HMAC-signed so API clients can keep using them across process reloads.
    payload = json.dumps({"u": username, "iat": int(time.time())}, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload)
    sig = hmac.new(SESSION_SIGNING_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    token = f"s1.{payload_b64}.{_b64url_encode(sig)}"
    sessions[token] = username
    return token


def _revoke_session_token(token: Optional[str]) -> None:
    if not token:
        return
    sessions.pop(token, None)
    revoked_sessions.add(token)


def _resolve_session_user(token: Optional[str]) -> Optional[str]:
    if not token or token in revoked_sessions:
        return None

    username = sessions.get(token)
    if username:
        return username

    claims = _decode_session_claims(token)
    if not claims:
        return None

    username = claims["u"]
    sessions[token] = username
    return username


def _extract_auth_token(
    query_token: Optional[str],
    bearer: Optional[HTTPAuthorizationCredentials],
    header_token: Optional[str],
    cookie_token: Optional[str],
) -> Optional[str]:
    if query_token:
        return query_token
    if bearer and bearer.credentials:
        return bearer.credentials.strip()
    if header_token:
        return header_token.strip()
    if cookie_token:
        return cookie_token
    return None


def _extract_auth_token_from_request(request: Request) -> Optional[str]:
    query_token = request.query_params.get("token")
    auth_header = (request.headers.get("authorization") or "").strip()
    bearer_token = None
    if auth_header.lower().startswith("bearer "):
        bearer_token = auth_header[7:].strip()

    header_token = request.headers.get("x-session-token")
    cookie_token = request.cookies.get("session_token")
    return _extract_auth_token(query_token, None if not bearer_token else HTTPAuthorizationCredentials(scheme="Bearer", credentials=bearer_token), header_token, cookie_token)


async def get_optional_auth_context(
    session_token: Optional[str] = Cookie(None),
    token: Optional[str] = Query(None, description="Session token query parameter"),
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token", description="Session token header"),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(session_auth_scheme),
) -> AuthContext:
    auth_token = _extract_auth_token(token, bearer, x_session_token, session_token)
    return AuthContext(token=auth_token, username=_resolve_session_user(auth_token))


async def require_auth_context(auth: AuthContext = Depends(get_optional_auth_context)) -> AuthContext:
    if not auth.username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return auth


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=SESSION_TTL_SECONDS,
        samesite="lax",
    )


def _google_reauth_response(message: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "reauth_required": True,
            "reauth_url": "/login?reauth=google",
            "message": message,
            "detail": detail,
        },
    )


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("No image provided")

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image. Upload a JPG or PNG file.")
    return frame


def _decode_base64_image(image_data: str) -> np.ndarray:
    if not image_data:
        raise ValueError("No image provided")

    try:
        encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
        return _decode_image_bytes(base64.b64decode(encoded))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Invalid base64 image data") from exc


async def _read_upload_frame(image: UploadFile) -> np.ndarray:
    return _decode_image_bytes(await image.read())


def _verify_face_frame(frame: np.ndarray) -> dict:
    faces = face_app.get(frame)
    if len(faces) == 0:
        return {"detected": False, "message": "No face detected"}

    test_emb = get_face_embedding(frame, faces[0])
    best_match, best_similarity = _find_best_face_match(test_emb)

    if best_match and best_similarity >= FACE_MATCH_THRESHOLD:
        face_detection_cache[best_match] = datetime.now()
        return {
            "detected": True,
            "username": best_match,
            "confidence": round(float(best_similarity) * 100, 1),
            "cache_duration": 240,
        }

    return {
        "detected": False,
        "message": "Unknown face",
        "confidence": round(float(best_similarity) * 100, 1),
    }


def _process_face_frame(frame: np.ndarray, detect_only: bool = False) -> dict:
    faces = face_app.get(frame)

    if len(faces) == 0:
        if detect_only:
            return {"detected": False, "face_count": 0, "message": "No face detected"}
        return {"error": "No face detected"}

    if detect_only:
        face0 = faces[0]
        x1, y1, x2, y2 = face0.bbox.astype(int)
        h, w = frame.shape[:2]
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        kps = face0.kps.tolist() if getattr(face0, "kps", None) is not None else None

        return {
            "detected": True,
            "face_count": len(faces),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
            "frame_size": [int(w), int(h)],
            "face_ratio": round((bbox_w * bbox_h) / float(max(1, w * h)), 6),
            "kps": kps,
        }

    embedding = get_face_embedding(frame, faces[0])
    return {"embedding": embedding.tolist()}


def _face_login_from_frame(frame: np.ndarray, response: Response) -> dict:
    faces = face_app.get(frame)

    if len(faces) == 0:
        return {"success": False, "message": "No face detected"}

    test_emb = get_face_embedding(frame, faces[0])
    best_match, best_similarity = _find_best_face_match(test_emb)

    if not best_match or best_similarity < FACE_MATCH_THRESHOLD:
        return {
            "success": False,
            "message": f"Face not recognized (confidence: {best_similarity*100:.1f}%)",
            "confidence": round(float(best_similarity) * 100, 1),
        }

    user = get_user_by_username(best_match)
    if not user:
        return {"success": False, "message": "User not found in database"}
    if not user.get("google_id"):
        return {
            "success": False,
            "message": "This face profile is not linked to Google Sign-In. Please sign in with Google first.",
        }

    token = _issue_session_token(best_match)
    _set_session_cookie(response, token)
    face_detection_cache[best_match] = datetime.now()
    print(f"[DEBUG] Face login successful: {best_match} ({best_similarity*100:.1f}% confidence)")

    return {
        "success": True,
        "token": token,
        "redirect_url": f"/?token={urllib.parse.quote(token, safe='')}",
        "model": ACTIVE_FACE_MODEL_NAME,
        "username": user["username"],
        "full_name": user["full_name"],
        "confidence": round(float(best_similarity) * 100, 1),
        "message": f"Welcome back, {user['full_name'].split()[0]}!",
    }


def _extract_text_from_gmail_payload(payload: Optional[dict]) -> str:
    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body") or {}
    body_data = body.get("data")
    if mime_type == "text/plain" and body_data:
        try:
            return base64.urlsafe_b64decode(body_data.encode("utf-8")).decode("utf-8", errors="replace")
        except Exception:
            return ""

    for part in payload.get("parts") or []:
        text = _extract_text_from_gmail_payload(part)
        if text:
            return text

    if body_data:
        try:
            return base64.urlsafe_b64decode(body_data.encode("utf-8")).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return ""


def _serialize_gmail_message(message: dict, include_body: bool = False) -> dict:
    payload = message.get("payload") or {}
    headers = {
        (header.get("name") or "").lower(): header.get("value", "")
        for header in payload.get("headers") or []
    }
    body_text = _extract_text_from_gmail_payload(payload).strip()

    data = {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "label_ids": message.get("labelIds", []),
        "snippet": message.get("snippet", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
    }
    if include_body:
        data["body"] = body_text or message.get("snippet", "")
    return data


def _resolve_face_enrollment_target(request: Request, explicit_username: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[dict], Optional[str]]:
    auth_token = _extract_auth_token_from_request(request)
    authenticated_username = _resolve_session_user(auth_token)
    requested_username = (explicit_username or "").strip() or None

    if authenticated_username and requested_username and authenticated_username != requested_username:
        return None, auth_token, None, "Provided username does not match the authenticated session."

    target_username = authenticated_username or requested_username
    if not target_username:
        return None, auth_token, None, "Provide a username or sign in first before enrolling a face."

    user = get_user_by_username(target_username)
    if not user:
        return None, auth_token, None, "User not found."

    return target_username, auth_token, user, None


async def _ensure_voice_services_ready() -> dict:
    if voice_service_state["ready"]:
        return dict(voice_service_state)

    async with voice_service_warm_lock:
        if voice_service_state["ready"]:
            return dict(voice_service_state)

        voice_service_state["loading"] = True
        voice_service_state["last_error"] = ""

        try:
            await asyncio.to_thread(_get_whisper_model)
            voice_service_state["stt_ready"] = True

            from app.services.tts_service import warm_tts, get_sentence_audio_bytes

            await asyncio.to_thread(warm_tts)
            voice_service_state["tts_ready"] = True

            # Run one short synthesis so the first real response does not block.
            await asyncio.to_thread(get_sentence_audio_bytes, "Voice services ready.")

            voice_service_state["ready"] = True
            print("[VOICE] STT and TTS are ready.")
        except Exception as exc:
            voice_service_state["ready"] = False
            voice_service_state["last_error"] = str(exc)
            print(f"[VOICE] Warmup failed: {exc}")
            raise
        finally:
            voice_service_state["loading"] = False

    return dict(voice_service_state)

# ── Auto-select OAuth broker based on OAUTH_METHOD ──
OAUTH_METHOD = os.getenv("OAUTH_METHOD", "vps").lower().strip()
if OAUTH_METHOD == "ngrok":
    OAUTH_BROKER_BASE_URL = os.getenv("NGROK_OAUTH_BROKER_URL", "").strip().rstrip("/")
    PUBLIC_BASE_URL = os.getenv("NGROK_OAUTH_BROKER_URL", "").strip().rstrip("/")
    GOOGLE_OAUTH_REDIRECT_URI = os.getenv("NGROK_OAUTH_REDIRECT_URI", "").strip()
elif OAUTH_METHOD == "vps":
    OAUTH_BROKER_BASE_URL = os.getenv("VPS_OAUTH_BROKER_URL", "").strip().rstrip("/")
    PUBLIC_BASE_URL = os.getenv("VPS_OAUTH_BROKER_URL", "").strip().rstrip("/")
    GOOGLE_OAUTH_REDIRECT_URI = os.getenv("VPS_OAUTH_REDIRECT_URI", "").strip()
else:
    OAUTH_BROKER_BASE_URL = ""
    PUBLIC_BASE_URL = ""
    GOOGLE_OAUTH_REDIRECT_URI = ""

print(f"[OAuth Config] Method: {OAUTH_METHOD} | Broker: {OAUTH_BROKER_BASE_URL}")

# Cross-device pairing state shared between PC and phone.
# {pair_token: {"status": "pending"|"complete", "intent": "register"|"login", ...}}
pair_sessions = {}


def _broker_enabled() -> bool:
    return bool(OAUTH_BROKER_BASE_URL)


def _broker_request(method: str, path: str, payload: Optional[dict] = None, timeout: int = 12) -> dict:
    if not _broker_enabled():
        raise RuntimeError("oauth broker not configured")

    url = f"{OAUTH_BROKER_BASE_URL}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"broker http {exc.code}: {detail}")
    except Exception as exc:
        raise RuntimeError(f"broker request failed: {exc}")


def _create_local_session_from_google_claim(claim: dict, intent: str) -> dict:
    profile = claim.get("profile") or {}
    tokens = claim.get("tokens") or {}

    google_id = profile.get("google_id")
    if not google_id:
        raise RuntimeError("broker claim missing google_id")

    create_google_user(
        google_id=google_id,
        email=profile.get("email", ""),
        full_name=profile.get("full_name", "Google User"),
        tokens=tokens,
    )

    user = get_user_by_google_id(google_id)
    if not user:
        raise RuntimeError("failed to resolve local user after broker claim")

    token = _issue_session_token(user["username"])

    has_face = user["username"] in face_users_db and len(face_users_db[user["username"]]) > 0

    # Re-auth should only refresh OAuth permissions and return to home.
    # It must not force users into face-enrollment flow.
    if intent == "reauth":
        redirect_pc = False
        dest = f"/?token={token}"
    else:
        redirect_pc = not has_face
        dest = f"/?token={token}" if has_face else f"/setup-face?token={token}"

    return {
        "session_token": token,
        "dest": dest,
        "redirect_pc": redirect_pc,
    }

@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    session_token: Optional[str] = Cookie(None),
    token: Optional[str] = None,
):
    auth_token = token or session_token
    if not _resolve_session_user(auth_token):
        redirect = RedirectResponse(url="/login", status_code=302)
        redirect.delete_cookie(key="session_token", path="/")
        return redirect

    if token and token != session_token:
        redirect = RedirectResponse(url="/", status_code=302)
        _set_session_cookie(redirect, token)
        return redirect

    return render_template("index.html", request)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, reauth: str = ""):
    reauth_mode = (reauth or "").strip().lower() == "google"
    pair_intent = "reauth" if reauth_mode else "login"

    pair_token = secrets.token_urlsafe(16)
    pair_entry = {"status": "pending", "intent": pair_intent}

    qr_url = ""
    if _broker_enabled():
        try:
            # Broker only needs an auth-capable flow; local app preserves reauth semantics.
            broker_intent = "login" if pair_intent == "reauth" else pair_intent
            broker_data = _broker_request("POST", "/pair/create", payload={"intent": broker_intent})
            pair_entry["broker_pair"] = broker_data.get("pair")
            qr_url = broker_data.get("mobile_url", "")
        except Exception as exc:
            print(f"[OAuthBroker] login pair create failed: {exc}")

    pair_sessions[pair_token] = pair_entry
    return render_template(
        "login.html",
        request,
        {"pair_token": pair_token, "qr_url": qr_url},
    )

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    pair_token = secrets.token_urlsafe(16)
    pair_entry = {"status": "pending", "intent": "register"}

    qr_url = ""
    if _broker_enabled():
        try:
            broker_data = _broker_request("POST", "/pair/create", payload={"intent": "register"})
            pair_entry["broker_pair"] = broker_data.get("pair")
            qr_url = broker_data.get("mobile_url", "")
        except Exception as exc:
            print(f"[OAuthBroker] register pair create failed: {exc}")

    pair_sessions[pair_token] = pair_entry
    return render_template(
        "register.html",
        request,
        {"pair_token": pair_token, "qr_url": qr_url},
    )

@app.get("/setup-face", response_class=HTMLResponse)
async def setup_face_page(
    request: Request,
    session_token: Optional[str] = Cookie(None),
    token: Optional[str] = None,
):
    auth_token = token or session_token
    if not _resolve_session_user(auth_token):
        redirect = RedirectResponse(url="/login", status_code=302)
        redirect.delete_cookie(key="session_token", path="/")
        return redirect

    response = render_template("setup_face.html", request)
    if token and token != session_token:
        _set_session_cookie(response, token)
    return response


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return render_template("admin.html", request)

@app.get("/api/admin/users")
async def admin_list_users():
    return get_all_users()

@app.get("/api/admin/face-list")
async def admin_face_list():
    """Return list of usernames that have face embeddings enrolled."""
    return list(face_users_db.keys())

@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int):
    # Look up username before deleting so we can clean face DB
    all_u = get_all_users()
    target = next((u for u in all_u if u["id"] == user_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    delete_user_by_id(user_id)
    # Remove face embeddings from memory + disk
    uname = target["username"]
    if uname in face_users_db:
        del face_users_db[uname]
        save_face_database(face_users_db)
    return {"ok": True}

@app.put("/api/admin/users/{user_id}")
async def admin_update_user_route(user_id: int, request: Request):
    body = await request.json()
    ok = admin_update_user(
        user_id=user_id,
        full_name=body.get("full_name"),
        email=body.get("email"),
        location=body.get("location"),
        interests=body.get("interests"),
    )
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}

@app.post("/api/register")
async def register_disabled():
    raise HTTPException(
        status_code=410,
        detail="Manual registration has been removed. Use Google Sign-In to create an account.",
    )


@app.post("/api/login")
async def login_disabled():
    raise HTTPException(
        status_code=410,
        detail="Manual login has been removed. Use Google Sign-In or face login instead.",
    )

@app.post("/api/logout")
async def logout(response: Response, auth: AuthContext = Depends(get_optional_auth_context)):
    _revoke_session_token(auth.token)
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out successfully"}


# ── Google OAuth endpoints ─────────────────────────────────────────────────

# Temporary store for OAuth state tokens  {state_token: "register"|"login"}
_oauth_states: dict[str, str] = {}


def _resolve_oauth_redirect_uri(request: Request) -> str:
    """Build callback URL from current host (used for PC browser flows)."""
    # If explicitly configured, always honor the public callback.
    # This is required for phone QR flows because Google blocks private/LAN callbacks.
    if GOOGLE_OAUTH_REDIRECT_URI:
        return GOOGLE_OAUTH_REDIRECT_URI

    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host")
    scheme = forwarded_proto or request.url.scheme
    host = forwarded_host or request.url.netloc

    # Google blocks OAuth callbacks on private/LAN IPs for web client flows.
    host_only = host.split(":", 1)[0]
    
    # If the user is on localhost, this is perfectly fine for Google
    if host_only in ("localhost", "127.0.0.1", "[::1]"):
        return f"{scheme}://{host}/auth/google/callback"

    # If it's a private IP (e.g., 192.168.x.x), we must raise an error to trigger the mobile fallback
    ip_obj = None
    try:
        ip_obj = ipaddress.ip_address(host_only)
    except ValueError:
        ip_obj = None

    if ip_obj and ip_obj.is_private and not ip_obj.is_loopback:
        raise ValueError("private_ip_callback_blocked")

    return f"{scheme}://{host}/auth/google/callback"

@app.get("/auth/google")
async def google_auth_start(request: Request, intent: str = "register", pair: str = ""):
    """Redirect the browser to Google's OAuth consent screen."""
    try:
        redirect_uri = _resolve_oauth_redirect_uri(request)
    except ValueError as exc:
        if str(exc) == "private_ip_callback_blocked" and pair:
            return RedirectResponse(
                url=f"/mobile-connect?pair={pair}&error=private_ip_callback",
                status_code=302,
            )
        raise HTTPException(
            status_code=400,
            detail=(
                "Google OAuth callback on private IP is blocked. "
                "Use localhost on PC or set GOOGLE_OAUTH_REDIRECT_URI to a public HTTPS callback."
            ),
        )

    state = secrets.token_urlsafe(16)
    try:
        auth_url, code_verifier = build_auth_url_with_verifier(state=state, redirect_uri=redirect_uri)
        _oauth_states[state] = {
            "intent": intent,
            "redirect_uri": redirect_uri,
            "pair": pair,
            "code_verifier": code_verifier,
        }
    except FileNotFoundError as exc:
        if pair:
            return RedirectResponse(
                url=f"/mobile-connect?pair={pair}&error=oauth_start_failed",
                status_code=302,
            )
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        print(f"[OAuth] auth start failed: {exc}")
        if pair:
            return RedirectResponse(
                url=f"/mobile-connect?pair={pair}&error=oauth_start_failed",
                status_code=302,
            )
        raise HTTPException(status_code=500, detail="Failed to start Google OAuth")
    return RedirectResponse(url=auth_url)


@app.get("/auth/google/callback")
async def google_auth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    response: Response = None,
):
    """Handle the redirect back from Google after the user grants permission."""
    if error:
        return RedirectResponse(url=f"/register?error={error}", status_code=302)
    if not code:
        return RedirectResponse(url="/register?error=no_code", status_code=302)

    # Retrieve intent and redirect_uri stored when the flow started
    state_data = _oauth_states.pop(state, None) if state else None
    if isinstance(state_data, dict):
        intent = state_data.get("intent", "register")
        redirect_uri = state_data.get("redirect_uri")
        code_verifier = state_data.get("code_verifier")
    else:
        # legacy / fallback
        intent = state_data
        redirect_uri = None
        code_verifier = None

    try:
        tokens, profile = exchange_code_for_tokens(code, redirect_uri=redirect_uri, code_verifier=code_verifier)
    except Exception as exc:
        print(f"[OAuth] Token exchange failed: {exc}")
        return RedirectResponse(url="/register?error=token_exchange_failed", status_code=302)

    google_id = profile["google_id"]

    # Create or update the user record in the DB
    is_new = False
    try:
        _, is_new = create_google_user(
            google_id=google_id,
            email=profile["email"],
            full_name=profile["full_name"],
            tokens=tokens,
        )
    except Exception as exc:
        print(f"[OAuth] DB error: {exc}")
        return RedirectResponse(url="/register?error=db_error", status_code=302)

    # Look up the username that was assigned
    user = get_user_by_google_id(google_id)
    if not user:
        return RedirectResponse(url="/register?error=user_not_found", status_code=302)

    # Create a session
    token = _issue_session_token(user["username"])

    # New users OR users without face enrolled go to face setup
    has_face = user["username"] in face_users_db and len(face_users_db[user["username"]]) > 0
    if intent == "reauth":
        dest = f"/?token={token}"
    else:
        dest = f"/?token={token}" if has_face else f"/setup-face?token={token}"

    # If this OAuth was triggered from a phone (pair flow),
    # store the session so the PC can pick it up.
    pair_token = state_data.get("pair", "") if isinstance(state_data, dict) else ""
    if pair_token and pair_token in pair_sessions:
        needs_face_setup = (not has_face) and intent != "reauth"
        pair_sessions[pair_token].update({
            "status": "complete",
            "session_token": token,
            "dest": dest,
            "redirect_pc": needs_face_setup,
        })

        mode = "face_on_pc" if needs_face_setup else "done_mobile"
        redirect = RedirectResponse(url=f"/pair-complete?mode={mode}", status_code=302)
        _set_session_cookie(redirect, token)
        return redirect

    redirect = RedirectResponse(url=dest, status_code=302)
    _set_session_cookie(redirect, token)
    return redirect

@app.get("/api/user")
async def get_current_user(auth: AuthContext = Depends(require_auth_context)):
    user = get_user_by_username(auth.username)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        **user,
        "session_token": auth.token,
        "auth_via": "bearer_or_token",
    }


@app.post("/api/user/context")
async def update_user_context(
    request: Request,
    auth: AuthContext = Depends(require_auth_context),
):
    """Best-effort update of user location/interests from the current device context."""
    payload = await request.json()
    provided_location = str(payload.get("location", "") or "").strip()
    provided_interests = str(payload.get("interests", "") or "").strip()
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")

    resolved_location = provided_location

    # If browser geolocation was sent, resolve it into a human-readable place
    # using the same weather provider already used by this project.
    if not resolved_location and latitude is not None and longitude is not None:
        try:
            lat = float(latitude)
            lon = float(longitude)
            api_key = "10428bba45b34ba8b4543622252612"
            reverse_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}"

            import httpx
            async with httpx.AsyncClient(timeout=8.0) as client:
                reverse_response = await client.get(reverse_url)
                reverse_data = reverse_response.json()

            if reverse_response.status_code == 200 and reverse_data.get("location"):
                loc = reverse_data["location"]
                city = (loc.get("name") or "").strip()
                region = (loc.get("region") or "").strip()
                country = (loc.get("country") or "").strip()
                parts = [p for p in [city, region, country] if p]
                if parts:
                    resolved_location = ", ".join(parts)
        except Exception:
            pass

    update_location = resolved_location if resolved_location else None
    update_interests = provided_interests if provided_interests else None
    if update_location or update_interests:
        update_user_preferences(auth.username, location=update_location, interests=update_interests)

    user = get_user_by_username(auth.username) or {}
    return {
        "ok": True,
        "location": user.get("location", ""),
        "interests": user.get("interests", ""),
    }


@app.get("/api/voice/readiness")
async def voice_readiness(auth: AuthContext = Depends(require_auth_context)):
    try:
        state = await _ensure_voice_services_ready()
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                **voice_service_state,
                "ready": False,
                "message": "Voice services failed to load",
            },
        )

    return {
        **state,
        "message": "Voice services ready",
        "model": ACTIVE_FACE_MODEL_NAME,
    }

@app.get("/pair-complete", response_class=HTMLResponse)
async def pair_complete_page(request: Request, mode: str = "face_on_pc"):
    return render_template("pair_complete.html", request, {"mode": mode})

@app.get("/mobile-connect", response_class=HTMLResponse)
async def mobile_connect(request: Request, pair: str = ""):
    """Mobile landing page where phone completes Google OAuth."""
    if not pair or pair not in pair_sessions:
        return HTMLResponse("<h3 style='font-family:sans-serif;padding:40px'>QR code expired. Please refresh the register page on your PC and scan again.</h3>")
    intent = pair_sessions[pair].get("intent", "register")
    return render_template(
        "mobile_pair.html",
        request,
        {"pair_token": pair, "intent": intent},
    )

@app.post("/api/pair-trigger/{pair_token}")
async def pair_trigger(pair_token: str):
    """Phone calls this to signal the PC to start Google OAuth."""
    if pair_token not in pair_sessions:
        raise HTTPException(status_code=404, detail="Invalid or expired pair token")
    pair_sessions[pair_token]["status"] = "triggered"
    return {"ok": True}

@app.get("/api/pair-status/{pair_token}")
async def pair_status(pair_token: str):
    """PC polls this to detect when phone has completed Google OAuth."""
    if pair_token not in pair_sessions:
        return {"status": "unknown"}

    entry = pair_sessions[pair_token]

    broker_pair = entry.get("broker_pair")
    if broker_pair and entry.get("status") != "complete":
        try:
            broker_status = _broker_request("GET", f"/pair/status/{broker_pair}")
            state = broker_status.get("status", "pending")

            if state == "complete":
                claim = _broker_request("POST", f"/pair/claim/{broker_pair}")
                local_session = _create_local_session_from_google_claim(claim, entry.get("intent", "register"))
                entry.update({
                    "status": "complete",
                    "session_token": local_session["session_token"],
                    "dest": local_session["dest"],
                    "redirect_pc": bool(local_session["redirect_pc"]),
                })
            elif state in ("expired", "unknown"):
                entry["status"] = state
        except Exception as exc:
            entry["broker_error"] = str(exc)

    resp = {"status": entry["status"]}
    if entry["status"] == "complete":
        resp["session_token"] = entry.get("session_token")
        resp["dest"] = entry.get("dest", "/")
        resp["redirect_pc"] = bool(entry.get("redirect_pc", False))
    if entry.get("broker_error"):
        resp["broker_error"] = entry.get("broker_error")
    return resp

@app.get("/api/local-url")
async def get_local_url(request: Request, path: str = "/register"):
    """Return the public/broker URL if available, otherwise fall back to localhost."""
    if not path.startswith("/"):
        path = f"/{path}"

    # When running locally (localhost/LAN), QR links should point back to this machine,
    # not the public broker URL, otherwise phone and PC can end up on different servers.
    req_host = request.headers.get("x-forwarded-host") or request.url.netloc
    host_only = req_host.split(":", 1)[0].strip("[]")
    prefer_local = host_only in ("localhost", "127.0.0.1", "::1")
    if not prefer_local:
        try:
            ip_obj = ipaddress.ip_address(host_only)
            prefer_local = ip_obj.is_private or ip_obj.is_loopback
        except ValueError:
            prefer_local = False

    # Use configured public URL only when request is already on a public host.
    if PUBLIC_BASE_URL and not prefer_local:
        return {"url": f"{PUBLIC_BASE_URL}{path}"}

    # Fallback: use local IP if no broker configured
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return {"url": f"http://{ip}:8000{path}"}

@app.post("/api/face/verify")
async def verify_face(request: Request):
    """Verify if a user is in front of the mirror using face recognition"""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"detected": False, "message": "Face recognition not available"}
    
    try:
        data = await request.json()
        frame = _decode_base64_image(data.get("image"))
        return _verify_face_frame(frame)
    
    except Exception as e:
        print(f"[DEBUG] Face verification error: {e}")
        return {"detected": False, "message": str(e)}


@app.post("/api/face/verify-upload")
async def verify_face_upload(image: UploadFile = File(..., description="Face image to verify")):
    """Swagger-friendly face verification with multipart image upload."""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"detected": False, "message": "Face recognition not available"}

    try:
        frame = await _read_upload_frame(image)
        return _verify_face_frame(frame)
    except Exception as e:
        print(f"[DEBUG] Face verification upload error: {e}")
        return {"detected": False, "message": str(e)}


@app.post("/api/face/process")
async def process_face(request: Request):
    """Process face image and return embedding for registration"""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"error": "Face recognition not available"}
    
    try:
        data = await request.json()
        frame = _decode_base64_image(data.get("image"))
        return _process_face_frame(frame, detect_only=bool(data.get("detect_only", False)))
    
    except Exception as e:
        print(f"[DEBUG] Face processing error: {e}")
        return {"error": str(e)}


@app.post("/api/face/process-upload")
async def process_face_upload(
    image: UploadFile = File(..., description="Face image to process"),
    detect_only: bool = Form(False),
):
    """Swagger-friendly face processing with multipart image upload."""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"error": "Face recognition not available"}

    try:
        frame = await _read_upload_frame(image)
        return _process_face_frame(frame, detect_only=detect_only)
    except Exception as e:
        print(f"[DEBUG] Face processing upload error: {e}")
        return {"error": str(e)}


@app.post("/api/face/login")
async def face_login(request: Request, response: Response):
    """Login using face recognition without credentials"""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "message": "Face recognition not available"}
    
    try:
        data = await request.json()
        frame = _decode_base64_image(data.get("image"))
        return _face_login_from_frame(frame, response)
    
    except Exception as e:
        print(f"[DEBUG] Face login error: {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/face/login-upload")
async def face_login_upload(
    response: Response,
    image: UploadFile = File(..., description="Face image for login"),
):
    """Swagger-friendly face login with multipart image upload."""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "message": "Face recognition not available"}

    try:
        frame = await _read_upload_frame(image)
        return _face_login_from_frame(frame, response)
    except Exception as e:
        print(f"[DEBUG] Face login upload error: {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/face/enroll")
async def face_enroll(
    request: Request,
    response: Response,
):
    """Enroll face using either an authenticated session or an explicit username."""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "message": "Face recognition not available"}
    data = await request.json()
    target_username, auth_token, user, error_message = _resolve_face_enrollment_target(request, data.get("username"))
    if error_message:
        return {"success": False, "message": error_message}
    images = data.get("images", [])  # list of base64 frames

    if not images:
        return {"success": False, "message": "No images provided"}

    embeddings = []
    for img_b64 in images:
        try:
            frame = _decode_base64_image(img_b64)
            faces = face_app.get(frame)
            if faces:
                emb = get_face_embedding(frame, faces[0])
                embeddings.append(emb)
        except Exception:
            continue

    if not embeddings:
        return {"success": False, "message": "No face detected in provided images"}

    embeddings = _prepare_enrollment_embeddings(embeddings)
    if not embeddings:
        return {"success": False, "message": "Captured photos were too inconsistent. Please try again."}

    face_users_db[target_username] = embeddings
    save_face_database(face_users_db)
    if auth_token:
        _set_session_cookie(response, auth_token)
    print(f"[DEBUG] Enrolled {len(embeddings)} face embeddings for {target_username}")
    return {
        "success": True,
        "embeddings_saved": len(embeddings),
        "username": target_username,
        "session_token": auth_token,
        "redirect_url": f"/?token={urllib.parse.quote(auth_token, safe='')}" if auth_token else "/",
        "model": ACTIVE_FACE_MODEL_NAME,
        "full_name": user.get("full_name", ""),
    }


@app.post("/api/face/enroll-upload")
async def face_enroll_upload(
    request: Request,
    response: Response,
    images: list[UploadFile] = File(..., description="Multiple face images for enrollment"),
    username: Optional[str] = Form(None, description="Username to enroll when no session token is provided"),
):
    """Swagger-friendly face enrollment with multipart image upload."""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "message": "Face recognition not available"}
    target_username, auth_token, user, error_message = _resolve_face_enrollment_target(request, username)
    if error_message:
        return {"success": False, "message": error_message}
    if not images:
        return {"success": False, "message": "No images provided"}

    embeddings = []
    for image in images:
        try:
            frame = await _read_upload_frame(image)
            faces = face_app.get(frame)
            if faces:
                embeddings.append(get_face_embedding(frame, faces[0]))
        except Exception:
            continue

    if not embeddings:
        return {"success": False, "message": "No face detected in provided images"}

    embeddings = _prepare_enrollment_embeddings(embeddings)
    if not embeddings:
        return {"success": False, "message": "Captured photos were too inconsistent. Please try again."}

    face_users_db[target_username] = embeddings
    save_face_database(face_users_db)
    if auth_token:
        _set_session_cookie(response, auth_token)
    print(f"[DEBUG] Enrolled {len(embeddings)} face embeddings for {target_username} via upload")

    return {
        "success": True,
        "embeddings_saved": len(embeddings),
        "username": target_username,
        "session_token": auth_token,
        "redirect_url": f"/?token={urllib.parse.quote(auth_token, safe='')}" if auth_token else "/",
        "model": ACTIVE_FACE_MODEL_NAME,
        "full_name": user.get("full_name", ""),
    }


@app.get("/api/face/check-cache")
async def check_face_cache(auth: AuthContext = Depends(require_auth_context)):
    """Check if user was recently detected (within last 4 minutes)"""
    if auth.username in face_detection_cache:
        last_seen = face_detection_cache[auth.username]
        time_diff = (datetime.now() - last_seen).total_seconds()
        
        # Cache valid for 4 minutes (240 seconds)
        if time_diff < 240:
            return {
                "cached": True,
                "username": auth.username,
                "seconds_remaining": int(240 - time_diff)
            }
    
    return {"cached": False, "message": "Face verification needed"}

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_token: Optional[str] = Cookie(None),
    token: Optional[str] = None,
):
    await websocket.accept()

    # Resolve authenticated user from existing sessions dict
    auth_token = token or session_token
    username = _resolve_session_user(auth_token)
    if not username:
        await websocket.send_json({"type": "error", "text": "Not authenticated"})
        await websocket.close()
        return
    user = get_user_by_username(username)  # from database.py
    if not user:
        await websocket.send_json({"type": "error", "text": "User not found"})
        await websocket.close()
        return

    # ── Set per-user Google credential context so calendar/gmail tools
    #    automatically use this user's personal OAuth tokens.
    try:
        import app.calendar_service as _cal_mod
        from app.services.gmail_service import set_current_user as _gmail_set_user
        from app.agent.tools import set_current_user as _agent_tools_set_user
        _cal_mod.set_current_user(username)
        _gmail_set_user(username)
        _agent_tools_set_user(username)
    except Exception:
        pass  # non-fatal if modules aren't loaded yet

    session_id = str(uuid.uuid4())

    # Import agent lazily to avoid circular imports and slow startup blocking
    from app.agent.graph import agent

    # Seed message history from DB — mirrors agent2_memory.py conversation_history pattern
    recent = get_recent_context(user['id'], limit=10)
    messages = []
    for m in recent:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    first_name = user['full_name'].split()[0]
    await websocket.send_json({
        "type": "status",
        "state": "ready",
        "text": f"Welcome back, {first_name}!"
    })

    # ── Speak a welcome greeting via TTS ──
    try:
        from app.services.tts_service import get_sentence_audio_bytes
        welcome_text = f"Welcome, {first_name}!"
        welcome_tts_started = time.perf_counter()
        welcome_wav = await asyncio.to_thread(get_sentence_audio_bytes, welcome_text)
        print(f"[TIMING] WS welcome_tts in {_elapsed_ms(welcome_tts_started):.0f} ms")
        if welcome_wav:
            import base64 as _b64
            await websocket.send_json({"type": "tts_audio", "data": _b64.b64encode(welcome_wav).decode('ascii')})
    except Exception as e:
        print(f"[WS] Welcome TTS error (non-fatal): {e}")

    try:
        def _is_google_reauth_hint(text: str) -> bool:
            lowered = (text or "").lower()
            markers = [
                "[reauth_required_google]",
                "no google oauth token found for this user",
                "missing gmail send permission",
                "insufficient authentication scopes",
                "insufficientpermissions",
                "invalid_grant",
                "google token is expired or revoked",
                "google authentication is incomplete or expired",
                "refresh token is missing",
                "credentials do not contain the necessary fields",
            ]
            return any(marker in lowered for marker in markers)

        while True:
            # Accept both text (JSON) and binary (audio) messages
            ws_message = await websocket.receive()

            if ws_message.get("type") == "websocket.disconnect":
                break

            turn_started = time.perf_counter()
            stt_ms = None
            first_token_ms = None
            tts_total_ms = 0.0
            request_kind = "text"
            user_text = None

            if "text" in ws_message:
                data = json.loads(ws_message["text"])
                if data.get("type") == "message":
                    user_text = data.get("text", "").strip()
                elif data.get("type") == "audio":
                    # Base64-encoded audio from browser
                    audio_b64 = data.get("data", "")
                    if audio_b64:
                        request_kind = "voice"
                        audio_bytes = base64.b64decode(audio_b64)
                        stt_started = time.perf_counter()
                        transcript = await asyncio.to_thread(transcribe_audio_bytes, audio_bytes)
                        stt_ms = _elapsed_ms(stt_started)
                        print(f"[TIMING] WS STT {stt_ms:.0f} ms ({len(audio_bytes)} bytes)")
                        if transcript:
                            print(f"[VOICE] \"{transcript}\"")
                            await websocket.send_json({"type": "transcript", "text": transcript})
                            user_text = transcript
                        else:
                            await websocket.send_json({"type": "voice_state", "state": "idle"})
                            await websocket.send_json({"type": "status", "text": "Could not understand audio", "state": "ready"})
                            continue

            elif "bytes" in ws_message:
                # Raw binary audio
                request_kind = "voice"
                audio_bytes = ws_message["bytes"]
                stt_started = time.perf_counter()
                transcript = await asyncio.to_thread(transcribe_audio_bytes, audio_bytes)
                stt_ms = _elapsed_ms(stt_started)
                print(f"[TIMING] WS STT {stt_ms:.0f} ms ({len(audio_bytes)} bytes)")
                if transcript:
                    print(f"[VOICE] \"{transcript}\"")
                    await websocket.send_json({"type": "transcript", "text": transcript})
                    user_text = transcript
                else:
                    await websocket.send_json({"type": "voice_state", "state": "idle"})
                    await websocket.send_json({"type": "status", "text": "Could not understand audio", "state": "ready"})
                    continue

            if not user_text:
                continue

            # ── "bye bye" → logout ──
            _bye_normalised = user_text.lower().strip().rstrip('.!?,')
            if _bye_normalised in ('bye bye', 'bye-bye', 'byebye', 'bye', 'goodbye', 'good bye', 'log out', 'logout', 'sign out'):
                try:
                    from app.services.tts_service import get_sentence_audio_bytes as _bye_tts
                    bye_text = f"Goodbye, {first_name}. See you later!"
                    bye_wav = await asyncio.to_thread(_bye_tts, bye_text)
                    if bye_wav:
                        await websocket.send_json({"type": "tts_audio", "data": base64.b64encode(bye_wav).decode('ascii')})
                except Exception:
                    pass
                await websocket.send_json({"type": "logout"})
                # Clean up server session
                _revoke_session_token(auth_token)
                break

            # Push thinking state to UI
            await websocket.send_json({"type": "voice_state", "state": "thinking"})
            await websocket.send_json({"type": "animation_start"})

            # Persist user message to conversation_history table
            save_conversation(user['id'], session_id, "user", user_text)

            # Append to in-memory history (same as agent2_memory.py pattern)
            messages.append(HumanMessage(content=user_text))

            agent_started = time.perf_counter()
            google_reauth_required = False
            try:
                # ── Streaming agent invocation ──────────────────────────
                # Stream tokens from the LLM and pipe completed sentences
                # to TTS immediately, so the user hears the first sentence
                # while the model is still generating the rest.
                import re as _re
                from app.services.tts_service import get_sentence_audio_bytes

                full_response = ""
                sentence_buffer = ""
                tts_tasks = []  # background TTS tasks (now produce audio bytes)
                sent_first_chunk = False
                final_result = None  # to capture tool messages for history
                _inside_think = False  # track <think> block state for streaming

                async for event in agent.astream_events(
                    {
                        "messages": messages,
                        "current_user": username,
                        "user_id": user['id'],
                        "session_id": session_id,
                        "user_location": user.get('news_country') or user.get('location', ''),
                        "user_interests": user.get('news_interests') or user.get('interests', ''),
                        "voice_state": "thinking",
                        "pending_confirmation": None,
                        "pending_action": None,
                        "draft_email": None,
                        "final_response": None,
                        "error": None,
                    },
                    version="v2",
                ):
                    kind = event.get("event")

                    # Deterministic tool output hook: catch auth errors even if
                    # the model paraphrases them later.
                    if kind == "on_tool_end":
                        tool_name = str(event.get("name", "") or "")
                        tool_output = str(event.get("data", {}).get("output", "") or "")
                        if _is_google_reauth_hint(tool_output):
                            google_reauth_required = True

                    # Capture token-by-token output from the LLM node
                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            token = chunk.content
                            # Gemini streaming returns list of parts; normalize to str
                            if isinstance(token, list):
                                parts = []
                                for part in token:
                                    if isinstance(part, str):
                                        parts.append(part)
                                    elif isinstance(part, dict):
                                        parts.append(part.get("text", ""))
                                    elif hasattr(part, "text"):
                                        parts.append(part.text)
                                token = "".join(parts)

                            # Track whether we're inside a <think> block
                            if "<think>" in token:
                                _inside_think = True
                            if _inside_think:
                                if "</think>" in token:
                                    # Extract any text after </think>
                                    token = token.split("</think>", 1)[1]
                                    _inside_think = False
                                else:
                                    continue  # skip tokens inside think block
                            if not token:
                                continue

                            full_response += token
                            sentence_buffer += token

                            # Send every token to frontend for instant display
                            await websocket.send_json({
                                "type": "response_chunk",
                                "token": token,
                                "first": not sent_first_chunk,
                            })
                            if first_token_ms is None:
                                first_token_ms = _elapsed_ms(agent_started)
                            sent_first_chunk = True

                            # When a sentence boundary is hit, dispatch to TTS
                            # immediately so speech starts while LLM continues
                            sentence_match = _re.search(r'[.!?]\s', sentence_buffer)
                            if sentence_match:
                                sentence_end = sentence_match.end()
                                sentence = sentence_buffer[:sentence_end].strip()
                                sentence_buffer = sentence_buffer[sentence_end:]
                                if sentence:
                                    await websocket.send_json({"type": "voice_state", "state": "speaking"})
                                    sentence_tts_started = time.perf_counter()
                                    audio_wav = await asyncio.to_thread(get_sentence_audio_bytes, sentence)
                                    tts_chunk_ms = _elapsed_ms(sentence_tts_started)
                                    tts_total_ms += tts_chunk_ms
                                    print(f"[TIMING] WS TTS sentence in {tts_chunk_ms:.0f} ms ({len(sentence)} chars)")
                                    if audio_wav:
                                        audio_b64 = base64.b64encode(audio_wav).decode('ascii')
                                        await websocket.send_json({"type": "tts_audio", "data": audio_b64})

                    # Capture the final state after the graph finishes
                    if kind == "on_chain_end" and event.get("name") == "LangGraph":
                        final_result = event.get("data", {}).get("output")

                # Flush any remaining text in the sentence buffer
                remaining = sentence_buffer.strip()
                # Also strip any lingering <think> blocks from full response
                full_response = _re.sub(r"<think>[\s\S]*?</think>", "", full_response).strip()
                if remaining:
                    remaining = _re.sub(r"<think>[\s\S]*?</think>", "", remaining).strip()
                    if remaining:
                        sentence_tts_started = time.perf_counter()
                        audio_wav = await asyncio.to_thread(get_sentence_audio_bytes, remaining)
                        tts_chunk_ms = _elapsed_ms(sentence_tts_started)
                        tts_total_ms += tts_chunk_ms
                        print(f"[TIMING] WS TTS tail in {tts_chunk_ms:.0f} ms ({len(remaining)} chars)")
                        if audio_wav:
                            audio_b64 = base64.b64encode(audio_wav).decode('ascii')
                            await websocket.send_json({"type": "tts_audio", "data": audio_b64})

                response_text = full_response if full_response else "I didn't get a response."

                # Persist assistant response
                save_conversation(
                    user['id'], session_id, "assistant", response_text,
                    agent_type="AARVIS"
                )

                # Update in-memory history for next turn
                from langchain_core.messages import ToolMessage
                result_messages = final_result.get("messages", messages) if final_result and isinstance(final_result, dict) else messages + [AIMessage(content=response_text)]
                cleaned_messages = []
                pending_tool_results = []
                for m in result_messages:
                    if isinstance(m, ToolMessage):
                        tool_name = getattr(m, 'name', 'tool')
                        tool_content = str(m.content or "")

                        if _is_google_reauth_hint(tool_content):
                            google_reauth_required = True
                            tool_content = tool_content.replace("[REAUTH_REQUIRED_GOOGLE]", "").strip()

                        pending_tool_results.append(f"[{tool_name} result: {tool_content}]")
                        continue
                    if isinstance(m, AIMessage) and getattr(m, 'tool_calls', None):
                        continue
                    if isinstance(m, AIMessage):
                        if pending_tool_results:
                            context = "\n".join(pending_tool_results)
                            cleaned_messages.append(AIMessage(content=f"{context}\n\n{m.content}"))
                            pending_tool_results = []
                        else:
                            cleaned_messages.append(m)
                    elif isinstance(m, HumanMessage):
                        cleaned_messages.append(m)
                messages = cleaned_messages

            except Exception as agent_err:
                print(f"[WS] Agent error: {agent_err}")
                response_text = "I'm sorry, I encountered an error processing your request. Please try again."

            agent_total_ms = _elapsed_ms(agent_started)
            turn_total_ms = _elapsed_ms(turn_started)
            stt_display_ms = 0.0 if stt_ms is None else stt_ms
            first_token_display_ms = 0.0 if first_token_ms is None else first_token_ms
            print(
                f"[TIMING] WS turn type={request_kind} "
                f"stt={stt_display_ms:.0f} ms "
                f"first_token={first_token_display_ms:.0f} ms "
                f"agent={agent_total_ms:.0f} ms "
                f"tts={tts_total_ms:.0f} ms "
                f"total={turn_total_ms:.0f} ms"
            )

            # Send final complete response + state reset
            await websocket.send_json({"type": "animation_stop"})
            if google_reauth_required:
                await websocket.send_json({
                    "type": "reauth_required",
                    "provider": "google",
                    "service": "gmail",
                    "reauth_url": "/login?reauth=google",
                })
            await websocket.send_json({"type": "response", "text": response_text})
            await websocket.send_json({"type": "voice_state", "state": "idle"})
            await websocket.send_json({"type": "status", "text": "Ready", "state": "ready"})

    except WebSocketDisconnect:
        print(f"[WS] {username} disconnected")


def _infer_news_country_code(location: str | None) -> str | None:
    """Infer NewsAPI country code from free-form location text."""
    if not location:
        return None

    lowered = location.lower()
    location_map = {
        "united states": "us",
        "usa": "us",
        "america": "us",
        "united kingdom": "gb",
        "uk": "gb",
        "england": "gb",
        "india": "in",
        "nepal": "np",
        "australia": "au",
        "canada": "ca",
        "france": "fr",
        "germany": "de",
        "japan": "jp",
        "singapore": "sg",
        "uae": "ae",
        "united arab emirates": "ae",
    }

    for token, code in location_map.items():
        if token in lowered:
            return code
    return None


NEWSAPI_TOP_HEADLINES_COUNTRIES = {
    "ar", "au", "at", "be", "br", "bg", "ca", "cn", "co", "cu", "cz", "eg", "fr", "de", "gr",
    "hk", "hu", "in", "id", "ie", "il", "it", "jp", "lv", "lt", "my", "mx", "ma", "nl", "nz",
    "ng", "no", "np", "ph", "pl", "pt", "ro", "ru", "sa", "rs", "sg", "sk", "si", "za", "kr", "se",
    "ch", "tw", "th", "tr", "ae", "ua", "gb", "us", "ve"
}

NEWS_COUNTRY_QUERY_KEYWORDS = {
    "np": "nepal",
}


def _resolve_country_query_text(country: str | None, country_code: str | None) -> str | None:
    """Resolve best-effort country text for keyword-based news queries."""
    raw = (country or "").strip()
    if raw and len(raw) > 2:
        return raw
    if country_code:
        return NEWS_COUNTRY_QUERY_KEYWORDS.get(country_code, raw or country_code)
    return raw or None


def _normalize_news_country(country: str | None, location: str | None = None) -> str | None:
    """Normalize a country input to a NewsAPI two-letter country code."""
    cleaned = (country or "").strip().lower()
    if cleaned:
        if len(cleaned) == 2 and cleaned.isalpha():
            return cleaned
        inferred = _infer_news_country_code(cleaned)
        if inferred:
            return inferred

    return _infer_news_country_code(location)


def _build_news_url(
    api_key: str,
    interests: str | None,
    location: str | None,
    country: str | None = None,
    mode: str = "personalized",
) -> str:
    """Build a localized NewsAPI URL using user interests and location."""
    if mode == "world":
        query = urllib.parse.quote_plus("world OR global")
        return (
            "https://newsapi.org/v2/everything"
            f"?q={query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
        )

    valid_categories = ["business", "entertainment", "health", "science", "sports", "technology"]
    interest_list = [i.strip().lower() for i in (interests or "").split(",") if i.strip()]
    primary_interest = interest_list[0] if interest_list else ""
    category = primary_interest if primary_interest in valid_categories else None
    country_code = _normalize_news_country(country, location)
    country_supported = bool(country_code and country_code in NEWSAPI_TOP_HEADLINES_COUNTRIES)
    country_query_text = _resolve_country_query_text(country, country_code)

    if category and country_supported:
        return f"https://newsapi.org/v2/top-headlines?country={country_code}&category={category}&apiKey={api_key}"

    if category and country_query_text:
        query_text = f"{category} {country_query_text} {location or ''}".strip()
        query = urllib.parse.quote_plus(query_text or category)
        return f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"

    if category:
        query_text = f"{category} {location or ''}".strip()
        query = urllib.parse.quote_plus(query_text or category)
        return f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"

    if country_supported and not primary_interest:
        return f"https://newsapi.org/v2/top-headlines?country={country_code}&apiKey={api_key}"

    if country_query_text and not primary_interest:
        query = urllib.parse.quote_plus(country_query_text)
        return f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"

    query_terms = [t for t in [primary_interest, country_query_text, location] if t]
    query = urllib.parse.quote_plus(" ".join(query_terms) if query_terms else "world")
    return f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"


def _build_google_news_rss_url(
    interests: str | None,
    location: str | None,
    country: str | None = None,
    mode: str = "personalized",
) -> str:
    """Build a Google News RSS URL without requiring an API key."""
    if mode == "world":
        return "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"

    interest_list = [i.strip() for i in (interests or "").split(",") if i.strip()]
    primary_interest = interest_list[0] if interest_list else ""
    country_code = _normalize_news_country(country, location)
    country_query_text = _resolve_country_query_text(country, country_code)
    query_terms = [t for t in [primary_interest, country_query_text, location] if t]
    query_text = " ".join(query_terms).strip() or "world"
    query = urllib.parse.quote_plus(query_text)
    return f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def _extract_rss_titles(xml_text: str, limit: int = 5) -> list[dict[str, str]]:
    """Parse RSS feed text and return title dictionaries for UI consumers."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    articles: list[dict[str, str]] = []

    for item in root.iter():
        if not item.tag.endswith("item"):
            continue

        title_text = ""
        for child in item:
            if child.tag.endswith("title") and child.text:
                title_text = child.text.strip()
                break

        if title_text:
            articles.append({"title": title_text})
            if len(articles) >= limit:
                break

    return articles


async def _fetch_news_articles(
    interests: str | None,
    location: str | None,
    country: str | None = None,
    mode: str = "personalized",
    limit: int = 5,
) -> list[dict[str, str]]:
    """Fetch news with NewsAPI first, then fallback to Google News RSS."""
    import httpx

    news_api_key = os.getenv("NEWS_API_KEY", "").strip() or "b47750eb5d3a45cda2f4542d117a42e8"

    seeded_articles: list[dict[str, str]] = []

    # Primary source: NewsAPI.
    news_url = _build_news_url(
        news_api_key,
        interests=interests,
        location=location,
        country=country,
        mode=mode,
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(news_url)

        if response.status_code == 200:
            payload = response.json()
            if payload.get("status") == "ok":
                articles = [
                    {"title": article["title"].strip()}
                    for article in payload.get("articles", [])
                    if article.get("title")
                ]
                if len(articles) >= limit:
                    return articles[:limit]
                seeded_articles = articles
            else:
                print(f"[News] NewsAPI error payload: {payload.get('code')} - {payload.get('message')}")
        else:
            print(f"[News] NewsAPI HTTP {response.status_code}")
    except Exception as exc:
        print(f"[News] NewsAPI request failed: {exc}")

    # Fallback source: Google News RSS.
    rss_url = _build_google_news_rss_url(
        interests=interests,
        location=location,
        country=country,
        mode=mode,
    )
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(rss_url)

        if response.status_code == 200 and response.text:
            rss_articles = _extract_rss_titles(response.text, limit=limit)
            if rss_articles:
                print("[News] Served headlines via Google News RSS fallback")
                if seeded_articles:
                    seen = {a.get("title", "").strip().lower() for a in seeded_articles}
                    for article in rss_articles:
                        title = article.get("title", "").strip()
                        if not title:
                            continue
                        lowered = title.lower()
                        if lowered in seen:
                            continue
                        seeded_articles.append({"title": title})
                        seen.add(lowered)
                        if len(seeded_articles) >= limit:
                            break
                    if seeded_articles:
                        return seeded_articles[:limit]
                return rss_articles[:limit]
        else:
            print(f"[News] RSS fallback HTTP {response.status_code}")
    except Exception as exc:
        print(f"[News] RSS fallback failed: {exc}")

    if seeded_articles:
        return seeded_articles[:limit]

    return []


def _format_calendar_events(events: list[dict]) -> list[dict]:
    formatted_events = []
    local_tz = datetime.now().astimezone().tzinfo
    now_local = datetime.now().astimezone()

    for event in events:
        start = event.get('start', {}).get('dateTime', event.get('start', {}).get('date'))
        end = event.get('end', {}).get('dateTime', event.get('end', {}).get('date'))
        summary = event.get('summary', 'Untitled Event')

        try:
            if start and 'T' in start:
                start_clean = start.replace('Z', '+00:00')
                end_clean = (end or start).replace('Z', '+00:00')

                try:
                    import re
                    event_time = datetime.fromisoformat(start_clean)
                    event_end_time = datetime.fromisoformat(end_clean)
                except Exception:
                    start_naive = re.sub(r'[+-]\\d{2}:\\d{2}$', '', start)
                    end_naive = re.sub(r'[+-]\\d{2}:\\d{2}$', '', end or start)
                    event_time = datetime.fromisoformat(start_naive.replace('Z', ''))
                    event_end_time = datetime.fromisoformat(end_naive.replace('Z', ''))

                if event_time.tzinfo is not None:
                    event_time_local = event_time.astimezone(local_tz)
                else:
                    event_time_local = event_time.replace(tzinfo=local_tz)

                if event_end_time.tzinfo is not None:
                    event_end_time_local = event_end_time.astimezone(local_tz)
                else:
                    event_end_time_local = event_end_time.replace(tzinfo=local_tz)

                formatted_events.append({
                    "id": event.get("id"),
                    "time": event_time_local.strftime("%I:%M %p"),
                    "endTime": event_end_time_local.strftime("%I:%M %p"),
                    "title": summary,
                    "status": "upcoming" if event_time_local > now_local else "past",
                    "startHour": event_time_local.hour + event_time_local.minute / 60,
                    "endHour": event_end_time_local.hour + event_end_time_local.minute / 60,
                    "start": start,
                    "end": end,
                    "description": event.get("description", ""),
                    "location": event.get("location", ""),
                })
            else:
                formatted_events.append({
                    "id": event.get("id"),
                    "time": "All Day",
                    "endTime": "All Day",
                    "title": summary,
                    "status": "all-day",
                    "startHour": 0,
                    "endHour": 24,
                    "start": start,
                    "end": end,
                    "description": event.get("description", ""),
                    "location": event.get("location", ""),
                })
        except Exception as e:
            print(f"[Calendar] Error parsing event '{summary}': {e}")
            formatted_events.append({
                "id": event.get("id"),
                "time": "Unknown",
                "endTime": "Unknown",
                "title": summary,
                "status": "unknown",
                "startHour": 0,
                "endHour": 1,
                "start": start,
                "end": end,
                "description": event.get("description", ""),
                "location": event.get("location", ""),
            })

    return formatted_events


def _calendar_reauth_payload(message: str, detail: str, extra: Optional[dict] = None) -> JSONResponse:
    content = {
        "reauth_required": True,
        "reauth_url": "/login?reauth=google",
        "message": message,
        "detail": detail,
    }
    if extra:
        content.update(extra)
    return JSONResponse(status_code=401, content=content)


def _ensure_calendar_user_context(username: str) -> None:
    try:
        set_calendar_current_user(username)
    except Exception:
        pass

@app.get("/api/weather")
async def get_weather(auth: AuthContext = Depends(get_optional_auth_context)):
    """Get weather data from WeatherAPI.com based on user's location"""
    import httpx
    
    # Get user's location preference
    location = "auto:ip"  # sensible fallback when user profile has no explicit location
    if auth.username:
        user = get_user_by_username(auth.username)
        if user and user.get('location'):
            location = user['location']
    
    API_KEY = "10428bba45b34ba8b4543622252612"
    # Use forecast endpoint to get min/max temps
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&days=1"
    
    try:
        weather_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            data = response.json()
        print(f"[TIMING] Weather upstream {location} in {_elapsed_ms(weather_started):.0f} ms")
        
        if response.status_code == 200:
            location_info = data.get("location") or {}
            resolved_location = ", ".join(
                [p for p in [location_info.get("name"), location_info.get("country")] if p]
            ) or location
            if 'forecast' in data and 'forecastday' in data['forecast']:
                forecast_day = data['forecast']['forecastday'][0]['day']
                return {
                    "temp": data['current']['temp_c'],
                    "condition": data['current']['condition']['text'],
                    "location": resolved_location,
                    "temp_min": forecast_day['mintemp_c'],
                    "temp_max": forecast_day['maxtemp_c']
                }
            else:
                return {
                    "temp": data['current']['temp_c'],
                    "condition": data['current']['condition']['text'],
                    "location": resolved_location,
                    "temp_min": 5,
                    "temp_max": 12
                }
        else:
            return {
                "temp": 8,
                "condition": "Unable to fetch weather",
                "location": location,
                "temp_min": 5,
                "temp_max": 12
            }
    except Exception as e:
        print(f"[Weather] Error: {e}")
        return {
            "temp": 8,
            "condition": "Connection timeout - check internet",
            "location": location,
            "temp_min": 5,
            "temp_max": 12
        }

@app.get("/api/news")
async def get_news(
    mode: str = Query(default="world", description="News mode: world or personalized"),
    interests: Optional[str] = Query(default=None, description="Optional personalized interests override"),
    country: Optional[str] = Query(default=None, description="Optional personalized country override"),
    auth: AuthContext = Depends(get_optional_auth_context),
):
    """Get top headlines in world mode (default) or personalized mode."""

    selected_mode = (mode or "world").strip().lower()
    if selected_mode not in {"world", "personalized"}:
        raise HTTPException(status_code=400, detail="mode must be 'world' or 'personalized'")

    selected_interests = (interests or "").strip() or None
    selected_country = (country or "").strip() or None
    selected_location = None

    if selected_mode == "personalized" and auth.username:
        saved = get_user_news_preferences(auth.username) or {}

        if not selected_interests:
            selected_interests = (
                (saved.get("news_interests") or "").strip()
                or (saved.get("legacy_interests") or "").strip()
                or None
            )

        if not selected_country:
            selected_country = (saved.get("news_country") or "").strip() or None

        selected_location = (saved.get("location") or "").strip() or None

    if selected_mode == "personalized" and not selected_interests and not selected_country and not selected_location:
        # No personalized settings are available yet, so return world headlines.
        selected_mode = "world"

    try:
        news_started = time.perf_counter()
        articles = await _fetch_news_articles(
            interests=selected_interests,
            location=selected_location,
            country=selected_country,
            mode=selected_mode,
            limit=5,
        )
        print(f"[TIMING] News upstream in {_elapsed_ms(news_started):.0f} ms")

        if articles:
            if selected_mode == "world" and len(articles) < 5:
                # Keep the dashboard shape stable with five slots.
                fillers = [
                    {"title": "More world headlines will appear shortly."},
                    {"title": "Global coverage is updating, please refresh soon."},
                    {"title": "News providers returned fewer headlines right now."},
                    {"title": "Additional world stories are currently unavailable."},
                ]
                for filler in fillers:
                    if len(articles) >= 5:
                        break
                    articles.append(filler)
            return articles

        if selected_mode == "world":
            return [
                {"title": "No world headlines available right now."},
                {"title": "Please check your connection and try again."},
                {"title": "The mirror will retry automatically."},
                {"title": "News providers may be temporarily unavailable."},
                {"title": "Please refresh in a few moments."},
            ]

        return [{"title": "No news available right now. Please try again shortly."}]
    except Exception as e:
        print(f"[News] Error: {e}")
        return [
            {"title": "Unable to connect to news service"},
            {"title": "Check your internet connection or firewall settings"},
            {"title": "The application will retry automatically"}
        ]


@app.get("/api/news/preferences")
async def get_news_preferences(auth: AuthContext = Depends(require_auth_context)):
    """Return saved and effective personalized-news preferences for the current user."""
    saved = get_user_news_preferences(auth.username)
    if not saved:
        raise HTTPException(status_code=404, detail="User not found")

    explicit_interests = (saved.get("news_interests") or "").strip()
    explicit_country = (saved.get("news_country") or "").strip()
    legacy_interests = (saved.get("legacy_interests") or "").strip()
    location = (saved.get("location") or "").strip()

    effective_interests = explicit_interests or legacy_interests
    effective_country = _normalize_news_country(explicit_country, location) or ""

    return {
        "interests": explicit_interests,
        "country": explicit_country,
        "effective_interests": effective_interests,
        "effective_country": effective_country,
        "location": location,
    }


@app.put("/api/news/preferences")
async def update_news_preferences(
    payload: NewsPreferenceUpdateRequest,
    auth: AuthContext = Depends(require_auth_context),
):
    """Save or clear personalized-news interests/country for the current user."""
    interests = payload.interests.strip() if payload.interests is not None else None
    country = payload.country.strip() if payload.country is not None else None

    if interests is None and country is None:
        raise HTTPException(status_code=400, detail="Provide interests and/or country")

    normalized_country = country
    if country is not None and country != "":
        normalized_country = _normalize_news_country(country)
        if not normalized_country:
            raise HTTPException(
                status_code=400,
                detail="country must be a valid country name or a 2-letter country code",
            )

    updated = update_user_news_preferences(
        auth.username,
        news_interests=interests,
        news_country=normalized_country,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    saved = get_user_news_preferences(auth.username) or {}
    return {
        "ok": True,
        "interests": (saved.get("news_interests") or "").strip(),
        "country": (saved.get("news_country") or "").strip(),
    }

@app.get("/api/calendar")
async def get_calendar(auth: AuthContext = Depends(get_optional_auth_context)):
    """Get today's events from Google Calendar"""
    if not CALENDAR_AVAILABLE:
        return {
            "events": [
                {"time": "09:00 AM", "endTime": "10:00 AM", "title": "Team Standup", "status": "upcoming", "startHour": 9.0, "endHour": 10.0},
                {"time": "02:00 PM", "endTime": "04:00 PM", "title": "Client Meeting", "status": "upcoming", "startHour": 14.0, "endHour": 16.0},
                {"time": "05:30 PM", "endTime": "06:30 PM", "title": "Gym Session", "status": "upcoming", "startHour": 17.5, "endHour": 18.5}
            ]
        }

    if not auth.username:
        return JSONResponse(
            status_code=401,
            content={
                "events": [],
                "reauth_required": True,
                "reauth_url": "/login?reauth=google",
                "message": "Session expired. Please sign in again.",
            },
        )

    _ensure_calendar_user_context(auth.username)

    try:
        calendar_started = time.perf_counter()
        events = get_todays_events(raise_on_auth_error=True)
        print(f"[TIMING] Calendar upstream in {_elapsed_ms(calendar_started):.0f} ms")

        if not events:
            return {"events": []}

        sorted_events = sorted(events, key=lambda e: e['start'].get('dateTime', e['start'].get('date')))
        return {"events": _format_calendar_events(sorted_events)}

    except GoogleReauthRequiredError as e:
        print(f"[Calendar] Re-auth required for {auth.username}: {e}")
        return _calendar_reauth_payload(
            "Google Calendar access expired. Please re-authenticate from your phone QR.",
            str(e),
            extra={"events": []},
        )
    except Exception as e:
        print(f"[Calendar] Error: {e}")
        return {"error": str(e), "events": []}


@app.get("/api/calendar/upcoming")
async def get_calendar_upcoming(
    max_results: int = Query(10, ge=1, le=50, description="Maximum number of upcoming events"),
    auth: AuthContext = Depends(require_auth_context),
):
    """Get upcoming Google Calendar events for the authenticated user."""
    _ensure_calendar_user_context(auth.username)

    try:
        events = get_upcoming_events(max_results=max_results, raise_on_auth_error=True)
        return {"events": _format_calendar_events(events), "count": len(events)}
    except GoogleReauthRequiredError as e:
        print(f"[Calendar] Upcoming re-auth required for {auth.username}: {e}")
        return _calendar_reauth_payload(
            "Google Calendar access expired. Please re-authenticate from your phone QR.",
            str(e),
            extra={"events": []},
        )
    except Exception as e:
        print(f"[Calendar] Upcoming error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch upcoming events: {e}")


@app.get("/api/calendar/range")
async def get_calendar_range(
    start: datetime = Query(..., description="Range start as ISO datetime"),
    end: datetime = Query(..., description="Range end as ISO datetime"),
    max_results: int = Query(50, ge=1, le=250, description="Maximum number of events"),
    auth: AuthContext = Depends(require_auth_context),
):
    """Get events in an arbitrary datetime range."""
    if end <= start:
        raise HTTPException(status_code=400, detail="'end' must be later than 'start'.")

    _ensure_calendar_user_context(auth.username)

    try:
        events = get_events_in_range(start, end, max_results=max_results, raise_on_auth_error=True)
        return {"events": _format_calendar_events(events), "count": len(events)}
    except GoogleReauthRequiredError as e:
        print(f"[Calendar] Range re-auth required for {auth.username}: {e}")
        return _calendar_reauth_payload(
            "Google Calendar access expired. Please re-authenticate from your phone QR.",
            str(e),
            extra={"events": []},
        )
    except Exception as e:
        print(f"[Calendar] Range error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch events in range: {e}")


@app.get("/api/calendar/events/{event_id}")
async def get_calendar_event_route(event_id: str, auth: AuthContext = Depends(require_auth_context)):
    """Get a single Google Calendar event by ID."""
    _ensure_calendar_user_context(auth.username)

    try:
        event = get_calendar_event(event_id, raise_on_auth_error=True)
        if not event:
            raise HTTPException(status_code=404, detail="Calendar event not found")
        return event
    except GoogleReauthRequiredError as e:
        print(f"[Calendar] Event fetch re-auth required for {auth.username}: {e}")
        return _calendar_reauth_payload(
            "Google Calendar access expired. Please re-authenticate from your phone QR.",
            str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Calendar] Event fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch calendar event: {e}")


@app.post("/api/calendar/events")
async def create_calendar_event_route(
    payload: CalendarEventCreateRequest,
    auth: AuthContext = Depends(require_auth_context),
):
    """Create a new Google Calendar event for the authenticated user."""
    if payload.end_time <= payload.start_time:
        raise HTTPException(status_code=400, detail="'end_time' must be later than 'start_time'.")

    _ensure_calendar_user_context(auth.username)

    try:
        event = create_calendar_event(
            summary=payload.summary,
            start_time=payload.start_time,
            end_time=payload.end_time,
            description=payload.description,
            location=payload.location,
            timezone_name=payload.timezone,
        )
        if not event:
            raise HTTPException(status_code=500, detail="Calendar event was not created.")
        return event
    except GoogleReauthRequiredError as e:
        print(f"[Calendar] Create re-auth required for {auth.username}: {e}")
        return _calendar_reauth_payload(
            "Google Calendar access expired. Please re-authenticate from your phone QR.",
            str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Calendar] Create error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not create calendar event: {e}")


@app.put("/api/calendar/events/{event_id}")
async def update_calendar_event_route(
    event_id: str,
    payload: CalendarEventUpdateRequest,
    auth: AuthContext = Depends(require_auth_context),
):
    """Update an existing Google Calendar event."""
    if payload.start_time and payload.end_time and payload.end_time <= payload.start_time:
        raise HTTPException(status_code=400, detail="'end_time' must be later than 'start_time'.")

    _ensure_calendar_user_context(auth.username)

    try:
        updated = update_calendar_event(
            event_id=event_id,
            summary=payload.summary,
            start_time=payload.start_time,
            end_time=payload.end_time,
            description=payload.description,
            location=payload.location,
            timezone_name=payload.timezone,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Calendar event not found or could not be updated")
        return updated
    except GoogleReauthRequiredError as e:
        print(f"[Calendar] Update re-auth required for {auth.username}: {e}")
        return _calendar_reauth_payload(
            "Google Calendar access expired. Please re-authenticate from your phone QR.",
            str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Calendar] Update error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not update calendar event: {e}")


@app.delete("/api/calendar/events/{event_id}")
async def delete_calendar_event_route(event_id: str, auth: AuthContext = Depends(require_auth_context)):
    """Delete a Google Calendar event."""
    _ensure_calendar_user_context(auth.username)

    try:
        deleted = delete_calendar_event(event_id, raise_on_auth_error=True)
        if not deleted:
            raise HTTPException(status_code=404, detail="Calendar event not found or could not be deleted")
        return {"ok": True, "event_id": event_id}
    except GoogleReauthRequiredError as e:
        print(f"[Calendar] Delete re-auth required for {auth.username}: {e}")
        return _calendar_reauth_payload(
            "Google Calendar access expired. Please re-authenticate from your phone QR.",
            str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Calendar] Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not delete calendar event: {e}")


@app.get("/api/mail/inbox")
async def mail_inbox(
    max_results: int = Query(10, ge=1, le=50, description="Maximum messages to fetch"),
    unread_only: bool = Query(True, description="Only return unread inbox messages"),
    auth: AuthContext = Depends(require_auth_context),
):
    """List Gmail inbox messages for the authenticated user."""
    try:
        from app.services.gmail_service import get_gmail_service

        service = get_gmail_service(username=auth.username)
        label_ids = ["INBOX"]
        if unread_only:
            label_ids.append("UNREAD")

        results = service.users().messages().list(
            userId="me",
            labelIds=label_ids,
            maxResults=max_results,
        ).execute()

        messages = []
        for item in results.get("messages", []):
            full_message = service.users().messages().get(
                userId="me",
                id=item["id"],
                format="full",
            ).execute()
            messages.append(_serialize_gmail_message(full_message))

        return {"messages": messages, "count": len(messages)}
    except GoogleReauthRequiredError as e:
        return _google_reauth_response(
            "Google Mail access expired. Please re-authenticate from your phone QR.",
            str(e),
        )
    except Exception as e:
        print(f"[Mail] Inbox error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch inbox: {e}")


@app.get("/api/mail/message/{message_id}")
async def mail_message(message_id: str, auth: AuthContext = Depends(require_auth_context)):
    """Fetch a single Gmail message with extracted text body."""
    try:
        from app.services.gmail_service import get_gmail_service

        service = get_gmail_service(username=auth.username)
        message = service.users().messages().get(
            userId="me",
            id=message_id,
            format="full",
        ).execute()
        return _serialize_gmail_message(message, include_body=True)
    except GoogleReauthRequiredError as e:
        return _google_reauth_response(
            "Google Mail access expired. Please re-authenticate from your phone QR.",
            str(e),
        )
    except Exception as e:
        print(f"[Mail] Message fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch email: {e}")


@app.post("/api/mail/send")
async def mail_send(payload: MailSendRequest, auth: AuthContext = Depends(require_auth_context)):
    """Send a plain-text Gmail message for the authenticated user."""
    if "@" not in payload.to:
        raise HTTPException(status_code=400, detail="Provide a valid recipient email address in 'to'.")

    subject = (payload.subject or payload.topic or "Message").strip()
    body = (payload.body or "").strip()
    additional_context = (payload.additional_context or "").strip()
    if not body:
        topic = (payload.topic or "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Provide either 'body' or 'topic'.")
        body_lines = [
            "Hello,",
            "",
            topic,
        ]
        if additional_context:
            body_lines.extend(["", additional_context])
        body_lines.extend(["", "Best regards,"])
        body = "\n".join(body_lines)
    elif additional_context:
        body = f"{body}\n\n{additional_context}"

    try:
        from app.services.gmail_service import get_gmail_service

        service = get_gmail_service(username=auth.username, require_send_scope=True)
        message = MIMEText(body)
        message["to"] = payload.to.strip()
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()

        return {
            "ok": True,
            "id": sent.get("id"),
            "thread_id": sent.get("threadId"),
            "to": payload.to.strip(),
            "subject": subject,
            "body": body,
        }
    except GoogleReauthRequiredError as e:
        return _google_reauth_response(
            "Google Mail send access is missing or expired. Please re-authenticate from your phone QR.",
            str(e),
        )
    except Exception as e:
        print(f"[Mail] Send error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not send email: {e}")


@app.post("/api/briefing/trigger")
async def trigger_briefing(auth: AuthContext = Depends(require_auth_context)):
    """
    Called by the frontend after successful face login.
    Generates a personalized morning briefing using Ollama + existing data sources.
    """
    user = get_user_by_username(auth.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    first_name = user['full_name'].split()[0]

    # Fetch calendar events using existing function
    from app.calendar_service import get_todays_events as briefing_get_events
    import httpx

    events = []
    try:
        set_calendar_current_user(auth.username)
        events = briefing_get_events()
    except Exception:
        pass

    events_text = (
        "\n".join([
            f"- {e.get('summary', 'Untitled')} at {e['start'].get('dateTime', 'All day')}"
            for e in events
        ])
        if events else "No events today."
    )

    # Fetch weather
    weather_text = "Weather unavailable."
    try:
        API_KEY = "10428bba45b34ba8b4543622252612"
        location = user.get('location') or 'auto:ip'
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&days=1"
            )
            wd = resp.json()
            weather_text = f"{wd['current']['temp_c']}°C, {wd['current']['condition']['text']}"
    except Exception:
        pass

    # Fetch news
    news_text = "News unavailable."
    try:
        interests = user.get('news_interests') or user.get('interests', 'technology')
        country = user.get('news_country') or None
        location = user.get('location') or ''
        news_articles = await _fetch_news_articles(
            interests=interests,
            location=location,
            country=country,
            mode="personalized",
            limit=3,
        )
        if news_articles:
            news_text = "; ".join([a["title"] for a in news_articles])
    except Exception:
        pass

    # Generate spoken briefing with Ollama
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="llama3:latest", temperature=0.5)
    prompt = (
        f"Generate a concise, friendly good morning briefing for {first_name}. "
        f"Keep it under 5 sentences — it will be read aloud.\n"
        f"Calendar today: {events_text}\n"
        f"Weather: {weather_text}\n"
        f"Top news: {news_text}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    briefing_text = response.content

    # Speak it via configured TTS service
    try:
        from app.services.tts_service import speak_async
        await speak_async(briefing_text)
    except Exception as tts_err:
        print(f"[Briefing] TTS error (non-fatal): {tts_err}")

    return {"briefing": briefing_text}


if __name__ == "__main__":
    import uvicorn
    import logging

    # Show INFO for our app, silence noisy library debug logs
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _noisy in (
        "httpcore", "httpx", "httpcore.connection", "httpcore.http11",
        "asyncio", "faster_whisper", "websockets", "uvicorn.error",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # Print LAN IP so user knows what to register in Google Console
    try:
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _lan_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        _lan_ip = "<could not detect>"
    print("\n" + "="*60)
    print("  Smart Mirror running at:")
    print(f"    PC   -> http://localhost:8000")
    print(f"    LAN  -> http://{_lan_ip}:8000")
    print(f"\n  OAuth redirect URI guidance:")
    print(f"    - Always keep: http://localhost:8000/auth/google/callback")
    print(f"    - For phone OAuth, use a PUBLIC HTTPS callback via GOOGLE_OAUTH_REDIRECT_URI")
    print(f"      (private/LAN IP callbacks are blocked by Google)")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def _print_lan_info():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "<could not detect>"
    print("\n" + "="*60)
    print(f"  Smart Mirror running at:")
    print(f"    PC   -> http://localhost:8000")
    print(f"    LAN  -> http://{lan_ip}:8000")
    print(f"\n  Google Console Redirect URIs:")
    print(f"    http://localhost:8000/auth/google/callback")
    print(f"    plus your public HTTPS callback if using phone OAuth")
    print("="*60 + "\n")
