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
from pydantic import BaseModel
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

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    # Optional dependency; environment variables can still be set by the shell.
    pass

from app.database import (
    create_user, verify_user, get_user_by_username, save_conversation, get_recent_context,
    create_google_user, get_user_by_google_id, get_user_google_tokens, update_google_tokens,
    get_all_users, delete_user_by_id, admin_update_user,
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
    from app.calendar_service import get_todays_events, get_upcoming_events
    CALENDAR_AVAILABLE = True
    print("[DEBUG] ✅ Calendar integration loaded successfully")
except Exception as e:
    CALENDAR_AVAILABLE = False
    print(f"[DEBUG] ❌ Calendar integration not available: {e}")

# Initialize face recognition (InsightFace for detection)
try:
    from insightface.app import FaceAnalysis
    print("[DEBUG] Initializing InsightFace...")
    face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    FACE_RECOGNITION_AVAILABLE = True
    print("[DEBUG] ✅ Face recognition ready!")
except Exception as e:
    FACE_RECOGNITION_AVAILABLE = False
    face_app = None
    print(f"[DEBUG] ❌ Face recognition not available: {e}")

# ── Custom Face Model ────────────────────────────────────────────────────────
try:
    import torch
    from torchvision import transforms
    from PIL import Image
    from app.ml.face_model import FaceEmbeddingModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[DEBUG] Activating Custom Nepali Face Backbone...")
    
    custom_face_model = FaceEmbeddingModel(embedding_size=512).to(device)
    custom_face_model.load_state_dict(torch.load(MODELS_DIR / "face_embedding_backbone.pth", map_location=device))
    custom_face_model.eval()
    
    custom_face_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    USE_CUSTOM_FACE_MODEL = True
    print("[DEBUG] ✅ Custom Face Model Loaded Successfully!")
except Exception as e:
    print(f"[DEBUG] ⚠️ Custom face model failed to load (will fallback to InsightFace): {e}")
    USE_CUSTOM_FACE_MODEL = False

def get_face_embedding(frame, face):
    """
    Given a raw BGR frame and an InsightFace detection object (`face`),
    generates a 512-D unit-norm embedding.
    Uses our custom fine-tuned PyTorch backbone if available, otherwise maps to InsightFace's default.
    """
    if USE_CUSTOM_FACE_MODEL:
        try:
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_crop = frame[y1:y2, x1:x2]
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            
            img_t = custom_face_transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                # Our backbone already outputs L2-normalized vectors
                emb = custom_face_model(img_t).cpu().numpy().squeeze()
            return emb
        except Exception as exc:
            print(f"[DEBUG] Custom face extraction error {exc}, falling back.")
            pass
            
    # Fallback to default InsightFace L2-normalized embedding
    return face.embedding / np.linalg.norm(face.embedding)

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


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    pad = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + pad)


def _issue_session_token(username: str) -> str:
    # Prefix helps distinguish signed stateless session tokens from old random tokens.
    payload = json.dumps({"u": username, "iat": int(time.time())}, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload)
    sig = hmac.new(SESSION_SIGNING_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    token = f"s1.{payload_b64}.{_b64url_encode(sig)}"
    sessions[token] = username
    return token


def _resolve_session_user(token: Optional[str]) -> Optional[str]:
    if not token:
        return None

    user = sessions.get(token)
    if user:
        return user

    if not token.startswith("s1."):
        return None

    try:
        _, payload_b64, sig_b64 = token.split(".", 2)
        expected_sig = hmac.new(
            SESSION_SIGNING_SECRET.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        actual_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        username = payload.get("u")
        issued = int(payload.get("iat", 0))
        if not username or issued <= 0:
            return None
        if int(time.time()) - issued > SESSION_TTL_SECONDS:
            return None
        return username
    except Exception:
        return None


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=SESSION_TTL_SECONDS,
        samesite="lax",
    )

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

# Pydantic models for request validation
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    location: str
    interests: str = ""
    face_embeddings: Optional[list] = None  # Face embeddings for registration

class LoginRequest(BaseModel):
    username: str
    password: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, session_token: Optional[str] = Cookie(None)):
    # Check if user is logged in
    if not _resolve_session_user(session_token):
        return RedirectResponse(url="/login", status_code=302)
    
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
async def setup_face_page(request: Request, session_token: Optional[str] = Cookie(None)):
    return templates.TemplateResponse("setup_face.html", {"request": request})


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
async def register(user: RegisterRequest, response: Response):
    try:
        user_id = create_user(
            username=user.username,
            email=user.email,
            password=user.password,
            full_name=user.full_name,
            location=user.location,
            interests=user.interests
        )
        
        # Save face embeddings if provided
        if user.face_embeddings and FACE_RECOGNITION_AVAILABLE:
            face_users_db[user.username] = user.face_embeddings
            save_face_database(face_users_db)
            print(f"[DEBUG] Saved {len(user.face_embeddings)} face embeddings for {user.username}")
        
        # Create session
        token = _issue_session_token(user.username)
        
        # Set cookie in response
        _set_session_cookie(response, token)
        
        return {
            "message": "User created successfully",
            "token": token,
            "username": user.username
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[DEBUG] Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/login")
async def login(credentials: LoginRequest, response: Response):
    user = verify_user(credentials.username, credentials.password)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create session
    token = _issue_session_token(credentials.username)
    
    # Set cookie in response
    _set_session_cookie(response, token)
    
    return {
        "message": "Login successful",
        "token": token,
        "username": user['username'],
        "full_name": user['full_name']
    }

@app.post("/api/logout")
async def logout(session_token: Optional[str] = Cookie(None)):
    if session_token and session_token in sessions:
        del sessions[session_token]
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
        
        # Check against all registered users
        best_match = None
        best_similarity = 0
        
        for username, embeddings in face_users_db.items():
            similarities = [np.dot(emb, test_emb) for emb in embeddings]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = username
        
        # Threshold for face recognition (40% similarity)
        if best_similarity > 0.4:
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
        
        # Check against all registered users
        best_match = None
        best_similarity = 0
        
        for username, embeddings in face_users_db.items():
            if not embeddings:
                continue
            similarities = [np.dot(emb, test_emb) for emb in embeddings]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = username
        
        # Threshold for face recognition (40% similarity)
        if best_similarity > 0.4:
            # Get user from database
            user = get_user_by_username(best_match)
            
            if not user:
                return {"success": False, "message": "User not found in database"}
            
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

    face_users_db[username] = embeddings
    save_face_database(face_users_db)
    # Ensure browser gets an authenticated cookie even when auth came via token query param.
    _set_session_cookie(response, auth_token)
    print(f"[DEBUG] Enrolled {len(embeddings)} face embeddings for {username}")
    return {"success": True, "embeddings_saved": len(embeddings)}

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
        welcome_wav = await asyncio.to_thread(get_sentence_audio_bytes, welcome_text)
        if welcome_wav:
            import base64 as _b64
            await websocket.send_json({"type": "tts_audio", "data": _b64.b64encode(welcome_wav).decode('ascii')})
    except Exception as e:
        print(f"[WS] Welcome TTS error (non-fatal): {e}")

    try:
        while True:
            # Accept both text (JSON) and binary (audio) messages
            ws_message = await websocket.receive()

            if ws_message.get("type") == "websocket.disconnect":
                break

            user_text = None

            if "text" in ws_message:
                data = json.loads(ws_message["text"])
                if data.get("type") == "message":
                    user_text = data.get("text", "").strip()
                elif data.get("type") == "audio":
                    # Base64-encoded audio from browser
                    audio_b64 = data.get("data", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        transcript = await asyncio.to_thread(transcribe_audio_bytes, audio_bytes)
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
                audio_bytes = ws_message["bytes"]
                transcript = await asyncio.to_thread(transcribe_audio_bytes, audio_bytes)
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
                                    audio_wav = await asyncio.to_thread(get_sentence_audio_bytes, sentence)
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
                        audio_wav = await asyncio.to_thread(get_sentence_audio_bytes, remaining)
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
                        pending_tool_results.append(f"[{tool_name} result: {m.content}]")
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

            # Send final complete response + state reset
            await websocket.send_json({"type": "animation_stop"})
            await websocket.send_json({"type": "response", "text": response_text})
            await websocket.send_json({"type": "voice_state", "state": "idle"})
            await websocket.send_json({"type": "status", "text": "Ready", "state": "ready"})

    except WebSocketDisconnect:
        print(f"[WS] {username} disconnected")

@app.get("/api/weather")
async def get_weather(session_token: Optional[str] = Cookie(None)):
    """Get weather data from WeatherAPI.com based on user's location"""
    import httpx
    
    # Get user's location preference
    location = "Kathmandu"  # Default
    if session_token and session_token in sessions:
        username = sessions[session_token]
        user = get_user_by_username(username)
        if user and user.get('location'):
            location = user['location']
    
    API_KEY = "10428bba45b34ba8b4543622252612"
    # Use forecast endpoint to get min/max temps
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&days=1"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            data = response.json()
            
            if response.status_code == 200:
                if 'forecast' in data and 'forecastday' in data['forecast']:
                    forecast_day = data['forecast']['forecastday'][0]['day']
                    return {
                        "temp": data['current']['temp_c'],
                        "condition": data['current']['condition']['text'],
                        "location": location,
                        "temp_min": forecast_day['mintemp_c'],
                        "temp_max": forecast_day['maxtemp_c']
                    }
                else:
                    return {
                        "temp": data['current']['temp_c'],
                        "condition": data['current']['condition']['text'],
                        "location": location,
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
    import httpx
    
    # Get user's interests
    interests = None
    if session_token and session_token in sessions:
        username = sessions[session_token]
        user = get_user_by_username(username)
        if user and user.get('interests'):
            interests = user['interests']
    
    API_KEY = "b47750eb5d3a45cda2f4542d117a42e8"
    
    # Build URL based on interests
    if interests:
        # Use first interest as category or search query
        interest_list = [i.strip().lower() for i in interests.split(',')]
        # Try category first (business, entertainment, health, science, sports, technology)
        category = interest_list[0] if interest_list[0] in ['business', 'entertainment', 'health', 'science', 'sports', 'technology'] else None
        if category:
            url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={API_KEY}"
        else:
            # Use as search query
            url = f"https://newsapi.org/v2/everything?q={interest_list[0]}&sortBy=publishedAt&apiKey={API_KEY}"
    else:
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            data = response.json()
            
            if data.get("status") == "ok" and data.get("articles"):
                articles = [{"title": article["title"]} for article in data["articles"][:5]]
                return articles
            else:
                return [{"title": "Unable to fetch news at this time"}]
    except Exception as e:
        print(f"[News] Error: {e}")
        return [
            {"title": "⚠️ Unable to connect to news service"},
            {"title": "Check your internet connection or firewall settings"},
            {"title": "The application will retry automatically"}
        ]

@app.get("/api/calendar")
async def get_calendar():
    """Get today's events from Google Calendar"""
    if not CALENDAR_AVAILABLE:
        return {
            "events": [
                {"time": "09:00 AM", "endTime": "10:00 AM", "title": "Team Standup", "status": "upcoming", "startHour": 9.0, "endHour": 10.0},
                {"time": "02:00 PM", "endTime": "04:00 PM", "title": "Client Meeting", "status": "upcoming", "startHour": 14.0, "endHour": 16.0},
                {"time": "05:30 PM", "endTime": "06:30 PM", "title": "Gym Session", "status": "upcoming", "startHour": 17.5, "endHour": 18.5}
            ]
        }

    try:
        events = get_todays_events()

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
        location = user.get('location', 'Kathmandu')
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
        category = interests.split(',')[0].strip().lower()
        API_KEY_NEWS = "b47750eb5d3a45cda2f4542d117a42e8"
        valid_cats = ['business', 'entertainment', 'health', 'science', 'sports', 'technology']
        if category in valid_cats:
            news_url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={API_KEY_NEWS}"
        else:
            news_url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY_NEWS}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(news_url)
            nd = resp.json()
            if nd.get("status") == "ok":
                news_text = "; ".join([a["title"] for a in nd.get("articles", [])[:3]])
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

    # Speak it via Kokoro TTS
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
