from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Cookie, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
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
    face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
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

# Session storage (in production, use Redis or database)
sessions = {}
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


def _issue_session_token(username: str) -> str:
    # Tokens stay server-owned so an application restart clears every active session.
    payload = json.dumps({"u": username, "iat": int(time.time())}, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload)
    sig = hmac.new(SESSION_SIGNING_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    token = f"s1.{payload_b64}.{_b64url_encode(sig)}"
    sessions[token] = username
    return token


def _resolve_session_user(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    return sessions.get(token)


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=SESSION_TTL_SECONDS,
        samesite="lax",
    )


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
    if intent == "register":
        # For register, force face setup on local machine if not enrolled yet.
        redirect_pc = not has_face
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

    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    pair_token = secrets.token_urlsafe(16)
    pair_entry = {"status": "pending", "intent": "login"}

    qr_url = ""
    if _broker_enabled():
        try:
            broker_data = _broker_request("POST", "/pair/create", payload={"intent": "login"})
            pair_entry["broker_pair"] = broker_data.get("pair")
            qr_url = broker_data.get("mobile_url", "")
        except Exception as exc:
            print(f"[OAuthBroker] login pair create failed: {exc}")

    pair_sessions[pair_token] = pair_entry
    return templates.TemplateResponse("login.html", {"request": request, "pair_token": pair_token, "qr_url": qr_url})

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
    return templates.TemplateResponse("register.html", {"request": request, "pair_token": pair_token, "qr_url": qr_url})

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

    response = templates.TemplateResponse("setup_face.html", {"request": request})
    if token and token != session_token:
        _set_session_cookie(response, token)
    return response


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

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
async def logout(response: Response, session_token: Optional[str] = Cookie(None)):
    if session_token and session_token in sessions:
        del sessions[session_token]
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
    dest = f"/?token={token}" if has_face else f"/setup-face?token={token}"

    # If this OAuth was triggered from a phone (pair flow),
    # store the session so the PC can pick it up.
    pair_token = state_data.get("pair", "") if isinstance(state_data, dict) else ""
    if pair_token and pair_token in pair_sessions:
        needs_face_setup = not has_face
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
async def get_current_user(session_token: Optional[str] = Cookie(None)):
    username = _resolve_session_user(session_token)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = get_user_by_username(username)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@app.post("/api/user/context")
async def update_user_context(request: Request, session_token: Optional[str] = Cookie(None)):
    """Best-effort update of user location/interests from the current device context."""
    username = _resolve_session_user(session_token)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")

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
        update_user_preferences(username, location=update_location, interests=update_interests)

    user = get_user_by_username(username) or {}
    return {
        "ok": True,
        "location": user.get("location", ""),
        "interests": user.get("interests", ""),
    }


@app.get("/api/voice/readiness")
async def voice_readiness(session_token: Optional[str] = Cookie(None)):
    username = _resolve_session_user(session_token)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")

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
    return templates.TemplateResponse("pair_complete.html", {"request": request, "mode": mode})

@app.get("/mobile-connect", response_class=HTMLResponse)
async def mobile_connect(request: Request, pair: str = ""):
    """Mobile landing page where phone completes Google OAuth."""
    if not pair or pair not in pair_sessions:
        return HTMLResponse("<h3 style='font-family:sans-serif;padding:40px'>QR code expired. Please refresh the register page on your PC and scan again.</h3>")
    intent = pair_sessions[pair].get("intent", "register")
    return templates.TemplateResponse("mobile_pair.html", {"request": request, "pair_token": pair, "intent": intent})

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
        image_data = data.get('image')
        
        if not image_data:
            return {"detected": False, "message": "No image provided"}
        
        # Decode base64 image
        img_data = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        faces = face_app.get(frame)
        
        if len(faces) == 0:
            return {"detected": False, "message": "No face detected"}
        
        # Get face embedding using custom backbone or fallback
        test_emb = get_face_embedding(frame, faces[0])
        
        best_match, best_similarity = _find_best_face_match(test_emb)

        if best_match and best_similarity >= FACE_MATCH_THRESHOLD:
            # Update detection cache
            face_detection_cache[best_match] = datetime.now()
            
            return {
                "detected": True,
                "username": best_match,
                "confidence": round(float(best_similarity) * 100, 1),
                "cache_duration": 240  # 4 minutes in seconds
            }
        else:
            return {
                "detected": False,
                "message": "Unknown face",
                "confidence": round(float(best_similarity) * 100, 1)
            }
    
    except Exception as e:
        print(f"[DEBUG] Face verification error: {e}")
        return {"detected": False, "message": str(e)}

@app.post("/api/face/process")
async def process_face(request: Request):
    """Process face image and return embedding for registration"""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"error": "Face recognition not available"}
    
    try:
        data = await request.json()
        image_data = data.get('image')
        detect_only = bool(data.get('detect_only', False))
        
        if not image_data:
            return {"error": "No image provided"}
        
        # Decode base64 image
        img_data = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
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

            # kps is typically [left_eye, right_eye, nose, left_mouth, right_mouth]
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
        
        # Get custom normalized embedding
        embedding = get_face_embedding(frame, faces[0])
        
        return {"embedding": embedding.tolist()}
    
    except Exception as e:
        print(f"[DEBUG] Face processing error: {e}")
        return {"error": str(e)}

@app.post("/api/face/login")
async def face_login(request: Request, response: Response):
    """Login using face recognition without credentials"""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "message": "Face recognition not available"}
    
    try:
        data = await request.json()
        image_data = data.get('image')
        
        if not image_data:
            return {"success": False, "message": "No image provided"}
        
        # Decode base64 image
        img_data = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        faces = face_app.get(frame)
        
        if len(faces) == 0:
            return {"success": False, "message": "No face detected"}
        
        # Get face embedding using custom model or fallback
        test_emb = get_face_embedding(frame, faces[0])
        
        best_match, best_similarity = _find_best_face_match(test_emb)

        if best_match and best_similarity >= FACE_MATCH_THRESHOLD:
            # Get user from database
            user = get_user_by_username(best_match)
            
            if not user:
                return {"success": False, "message": "User not found in database"}
            if not user.get("google_id"):
                return {
                    "success": False,
                    "message": "This face profile is not linked to Google Sign-In. Please sign in with Google first."
                }
            
            # Create session (same as normal login)
            token = _issue_session_token(best_match)
            
            # Set cookie in response
            _set_session_cookie(response, token)
            
            # Update face detection cache
            face_detection_cache[best_match] = datetime.now()
            
            print(f"[DEBUG] Face login successful: {best_match} ({best_similarity*100:.1f}% confidence)")
            
            return {
                "success": True,
                "token": token,
                "redirect_url": f"/?token={urllib.parse.quote(token, safe='')}",
                "model": ACTIVE_FACE_MODEL_NAME,
                "username": user['username'],
                "full_name": user['full_name'],
                "confidence": round(float(best_similarity) * 100, 1),
                "message": f"Welcome back, {user['full_name'].split()[0]}!"
            }
        else:
            return {
                "success": False,
                "message": f"Face not recognized (confidence: {best_similarity*100:.1f}%)",
                "confidence": round(float(best_similarity) * 100, 1)
            }
    
    except Exception as e:
        print(f"[DEBUG] Face login error: {e}")
        return {"success": False, "message": str(e)}

@app.post("/api/face/enroll")
async def face_enroll(
    request: Request,
    response: Response,
    session_token: Optional[str] = Cookie(None),
    token: Optional[str] = None,
):
    """Enroll face for an already-logged-in user (e.g. after Google signup)."""
    # Prefer explicit query token (fresh from OAuth redirect) over potentially stale cookie.
    auth_token = token or session_token
    username = _resolve_session_user(auth_token)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "message": "Face recognition not available"}
    data = await request.json()
    images = data.get("images", [])  # list of base64 frames

    if not images:
        return {"success": False, "message": "No images provided"}

    embeddings = []
    for img_b64 in images:
        try:
            img_data = base64.b64decode(img_b64.split(',')[1] if ',' in img_b64 else img_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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

    face_users_db[username] = embeddings
    save_face_database(face_users_db)
    # Ensure browser gets an authenticated cookie even when auth came via token query param.
    _set_session_cookie(response, auth_token)
    print(f"[DEBUG] Enrolled {len(embeddings)} face embeddings for {username}")
    return {
        "success": True,
        "embeddings_saved": len(embeddings),
        "session_token": auth_token,
        "redirect_url": f"/?token={urllib.parse.quote(auth_token, safe='')}",
        "model": ACTIVE_FACE_MODEL_NAME,
    }

@app.get("/api/face/check-cache")
async def check_face_cache(session_token: Optional[str] = Cookie(None)):
    """Check if user was recently detected (within last 4 minutes)"""
    username = _resolve_session_user(session_token)
    if not username:
        return {"cached": False, "message": "Not authenticated"}
    
    if username in face_detection_cache:
        last_seen = face_detection_cache[username]
        time_diff = (datetime.now() - last_seen).total_seconds()
        
        # Cache valid for 4 minutes (240 seconds)
        if time_diff < 240:
            return {
                "cached": True,
                "username": username,
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
        _cal_mod.set_current_user(username)
        _gmail_set_user(username)
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
                "missing gmail send permission",
                "insufficient authentication scopes",
                "insufficientpermissions",
                "google token is expired or revoked",
                "google authentication is incomplete or expired",
                "refresh token is missing",
                "credentials do not contain the necessary fields",
                "mail server connection failed",
                "couldn't send the email",
                "could not send the email",
                "send manually",
                "please re-authenticate",
                "reauthenticate your gmail account",
                "reauthenticate your google account",
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
                if auth_token and auth_token in sessions:
                    del sessions[auth_token]
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
                        "user_location": user.get('location', 'Kathmandu'),
                        "user_interests": user.get('interests', 'technology'),
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
                        elif "send_email" in tool_name.lower() and "email failed to send" in tool_output.lower():
                            # Fallback: force reconnect flow on explicit send_email failures
                            # so the mirror recovers through QR instead of staying stuck.
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
                if _is_google_reauth_hint(response_text):
                    google_reauth_required = True

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


def _build_news_url(api_key: str, interests: str | None, location: str | None) -> str:
    """Build a localized NewsAPI URL using user interests and location."""
    valid_categories = ["business", "entertainment", "health", "science", "sports", "technology"]
    interest_list = [i.strip().lower() for i in (interests or "").split(",") if i.strip()]
    primary_interest = interest_list[0] if interest_list else ""
    category = primary_interest if primary_interest in valid_categories else None
    country_code = _infer_news_country_code(location)

    if category and country_code:
        return f"https://newsapi.org/v2/top-headlines?country={country_code}&category={category}&apiKey={api_key}"

    if category:
        query_text = f"{category} {location or ''}".strip()
        query = urllib.parse.quote_plus(query_text or category)
        return f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"

    if country_code and not primary_interest:
        return f"https://newsapi.org/v2/top-headlines?country={country_code}&apiKey={api_key}"

    query_terms = [t for t in [primary_interest, location] if t]
    query = urllib.parse.quote_plus(" ".join(query_terms) if query_terms else "world")
    return f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"


def _build_google_news_rss_url(interests: str | None, location: str | None) -> str:
    """Build a Google News RSS URL without requiring an API key."""
    interest_list = [i.strip() for i in (interests or "").split(",") if i.strip()]
    primary_interest = interest_list[0] if interest_list else ""
    query_terms = [t for t in [primary_interest, location] if t]
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


async def _fetch_news_articles(interests: str | None, location: str | None, limit: int = 5) -> list[dict[str, str]]:
    """Fetch news with NewsAPI first, then fallback to Google News RSS."""
    import httpx

    news_api_key = os.getenv("NEWS_API_KEY", "").strip() or "b47750eb5d3a45cda2f4542d117a42e8"

    # Primary source: NewsAPI.
    news_url = _build_news_url(news_api_key, interests=interests, location=location)
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
                if articles:
                    return articles[:limit]
            else:
                print(f"[News] NewsAPI error payload: {payload.get('code')} - {payload.get('message')}")
        else:
            print(f"[News] NewsAPI HTTP {response.status_code}")
    except Exception as exc:
        print(f"[News] NewsAPI request failed: {exc}")

    # Fallback source: Google News RSS.
    rss_url = _build_google_news_rss_url(interests=interests, location=location)
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(rss_url)

        if response.status_code == 200 and response.text:
            rss_articles = _extract_rss_titles(response.text, limit=limit)
            if rss_articles:
                print("[News] Served headlines via Google News RSS fallback")
                return rss_articles
        else:
            print(f"[News] RSS fallback HTTP {response.status_code}")
    except Exception as exc:
        print(f"[News] RSS fallback failed: {exc}")

    return []

@app.get("/api/weather")
async def get_weather(session_token: Optional[str] = Cookie(None)):
    """Get weather data from WeatherAPI.com based on user's location"""
    import httpx
    
    # Get user's location preference
    location = "auto:ip"  # sensible fallback when user profile has no explicit location
    if session_token and session_token in sessions:
        username = sessions[session_token]
        user = get_user_by_username(username)
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
async def get_news(session_token: Optional[str] = Cookie(None)):
    """Get top headlines from NewsAPI.org based on user interests"""

    # Get user's interests and location
    interests = None
    location = None
    if session_token and session_token in sessions:
        username = sessions[session_token]
        user = get_user_by_username(username)
        if user and user.get('interests'):
            interests = user['interests']
        if user and user.get('location'):
            location = user['location']

    try:
        news_started = time.perf_counter()
        articles = await _fetch_news_articles(interests=interests, location=location, limit=5)
        print(f"[TIMING] News upstream in {_elapsed_ms(news_started):.0f} ms")

        if articles:
            return articles

        return [{"title": "No news available right now. Please try again shortly."}]
    except Exception as e:
        print(f"[News] Error: {e}")
        return [
            {"title": "Unable to connect to news service"},
            {"title": "Check your internet connection or firewall settings"},
            {"title": "The application will retry automatically"}
        ]

@app.get("/api/calendar")
async def get_calendar(session_token: Optional[str] = Cookie(None)):
    """Get today's events from Google Calendar"""
    if not CALENDAR_AVAILABLE:
        return {
            "events": [
                {"time": "09:00 AM", "endTime": "10:00 AM", "title": "Team Standup", "status": "upcoming", "startHour": 9.0, "endHour": 10.0},
                {"time": "02:00 PM", "endTime": "04:00 PM", "title": "Client Meeting", "status": "upcoming", "startHour": 14.0, "endHour": 16.0},
                {"time": "05:30 PM", "endTime": "06:30 PM", "title": "Gym Session", "status": "upcoming", "startHour": 17.5, "endHour": 18.5}
            ]
        }

    username = _resolve_session_user(session_token)
    if not username:
        return JSONResponse(
            status_code=401,
            content={
                "events": [],
                "reauth_required": True,
                "reauth_url": "/login?reauth=google",
                "message": "Session expired. Please sign in again.",
            },
        )

    try:
        set_calendar_current_user(username)
    except Exception:
        pass

    try:
        calendar_started = time.perf_counter()
        events = get_todays_events(raise_on_auth_error=True)
        print(f"[TIMING] Calendar upstream in {_elapsed_ms(calendar_started):.0f} ms")

        if not events:
            return {"events": []}

        sorted_events = sorted(events, key=lambda e: e['start'].get('dateTime', e['start'].get('date')))

        formatted_events = []
        local_tz = datetime.now().astimezone().tzinfo
        now_local = datetime.now().astimezone()

        for event in sorted_events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            summary = event.get('summary', 'Untitled Event')

            try:
                if 'T' in start:  # dateTime format
                    start_clean = start.replace('Z', '+00:00')
                    end_clean = end.replace('Z', '+00:00')

                    try:
                        import re
                        event_time = datetime.fromisoformat(start_clean)
                        event_end_time = datetime.fromisoformat(end_clean)
                    except Exception:
                        start_naive = re.sub(r'[+-]\\d{2}:\\d{2}$', '', start)
                        end_naive = re.sub(r'[+-]\\d{2}:\\d{2}$', '', end)
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

                    time_str = event_time_local.strftime("%I:%M %p")
                    end_time_str = event_end_time_local.strftime("%I:%M %p")
                    status = "upcoming" if event_time_local > now_local else "past"

                    formatted_events.append({
                        "time": time_str,
                        "endTime": end_time_str,
                        "title": summary,
                        "status": status,
                        "startHour": event_time_local.hour + event_time_local.minute / 60,
                        "endHour": event_end_time_local.hour + event_end_time_local.minute / 60
                    })
                else:  # all-day event
                    formatted_events.append({
                        "time": "All Day",
                        "endTime": "All Day",
                        "title": summary,
                        "status": "all-day",
                        "startHour": 0,
                        "endHour": 24
                    })
            except Exception as e:
                print(f"[Calendar] Error parsing event '{summary}': {e}")
                formatted_events.append({
                    "time": "Unknown",
                    "endTime": "Unknown",
                    "title": summary,
                    "status": "unknown",
                    "startHour": 0,
                    "endHour": 1
                })

        return {"events": formatted_events}

    except GoogleReauthRequiredError as e:
        print(f"[Calendar] Re-auth required for {username}: {e}")
        return JSONResponse(
            status_code=401,
            content={
                "events": [],
                "reauth_required": True,
                "reauth_url": "/login?reauth=google",
                "message": "Google Calendar access expired. Please re-authenticate from your phone QR.",
                "detail": str(e),
            },
        )
    except Exception as e:
        print(f"[Calendar] Error: {e}")
        return {"error": str(e), "events": []}


@app.post("/api/briefing/trigger")
async def trigger_briefing(session_token: Optional[str] = Cookie(None)):
    """
    Called by the frontend after successful face login.
    Generates a personalized morning briefing using Ollama + existing data sources.
    """
    if not session_token or session_token not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")

    username = sessions[session_token]
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    first_name = user['full_name'].split()[0]

    # Fetch calendar events using existing function
    from app.calendar_service import get_todays_events as briefing_get_events
    import httpx

    events = []
    try:
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
        interests = user.get('interests', 'technology')
        location = user.get('location') or ''
        news_articles = await _fetch_news_articles(interests=interests, location=location, limit=3)
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
