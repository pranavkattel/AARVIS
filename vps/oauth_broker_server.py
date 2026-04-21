from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone
import secrets
import os
import time
import json
import base64
import hmac
import hashlib

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from services.google_oauth import build_auth_url_with_verifier, exchange_code_for_tokens

app = FastAPI(title="AARVIS OAuth Broker")


def _safe_pair_ttl_seconds() -> int:
    """Parse and clamp pair/state TTL to avoid accidental immediate expiry."""
    raw = (os.getenv("OAUTH_PAIR_TTL_SECONDS", "600") or "600").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 600
    # Keep OAuth windows practical while preventing pathological config values.
    return max(120, min(value, 3600))


PAIR_TTL_SECONDS = _safe_pair_ttl_seconds()

# ── Auto-select OAuth broker configuration based on OAUTH_METHOD ──
OAUTH_METHOD = os.getenv("OAUTH_METHOD", "vps").lower().strip()
if OAUTH_METHOD == "ngrok":
    PUBLIC_BASE_URL = os.getenv("NGROK_OAUTH_BROKER_URL", "").strip().rstrip("/")
    GOOGLE_OAUTH_REDIRECT_URI = os.getenv("NGROK_OAUTH_REDIRECT_URI", "").strip()
elif OAUTH_METHOD == "vps":
    PUBLIC_BASE_URL = os.getenv("VPS_OAUTH_BROKER_URL", "").strip().rstrip("/")
    GOOGLE_OAUTH_REDIRECT_URI = os.getenv("VPS_OAUTH_REDIRECT_URI", "").strip()
else:
    PUBLIC_BASE_URL = ""
    GOOGLE_OAUTH_REDIRECT_URI = ""

print(f"[OAuth Broker Config] Method: {OAUTH_METHOD} | Public Base: {PUBLIC_BASE_URL}")

# {pair_token: {status, intent, expires_at, profile, tokens, claimed}}
pair_sessions: dict[str, dict] = {}
# {state_token: {payload, expires_at}}
oauth_states: dict[str, dict] = {}
STATE_SIGNING_SECRET = os.getenv("OAUTH_STATE_SECRET", os.getenv("SECRET_KEY", "change-me-oauth-state-secret"))


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    pad = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + pad)


def _sign_state(payload: dict) -> str:
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_bytes)
    sig = hmac.new(STATE_SIGNING_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def _decode_state(state_token: str) -> Optional[dict]:
    try:
        payload_b64, sig_b64 = state_token.split(".", 1)
    except ValueError:
        return None

    expected_sig = hmac.new(
        STATE_SIGNING_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    actual_sig = _b64url_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None

    issued = int(payload.get("ts", 0))
    now = int(time.time())
    if issued <= 0 or (now - issued) > PAIR_TTL_SECONDS:
        return None

    return payload


def _decode_state_allow_expired(state_token: str) -> Optional[dict]:
    """Decode signed state while ignoring timestamp expiry (for retry recovery only)."""
    try:
        payload_b64, sig_b64 = state_token.split(".", 1)
    except ValueError:
        return None

    expected_sig = hmac.new(
        STATE_SIGNING_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception:
        return None

    if not hmac.compare_digest(expected_sig, actual_sig):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None


def _ensure_pair_entry(pair: str, intent: str = "register") -> dict:
    """Recreate an expired/missing pair entry so OAuth can continue safely."""
    if pair in pair_sessions:
        return pair_sessions[pair]

    expires_at = _utc_now() + timedelta(seconds=PAIR_TTL_SECONDS)
    pair_sessions[pair] = {
        "status": "pending",
        "intent": intent if intent in ("register", "login") else "register",
        "expires_at": expires_at,
        "profile": None,
        "tokens": None,
        "claimed": False,
    }
    return pair_sessions[pair]


def _create_pair_session(intent: str = "register") -> tuple[str, datetime]:
    """Create and store a fresh pair session for mobile OAuth continuation."""
    normalized_intent = intent if intent in ("register", "login") else "register"
    pair = secrets.token_urlsafe(24)
    expires_at = _utc_now() + timedelta(seconds=PAIR_TTL_SECONDS)
    pair_sessions[pair] = {
        "status": "pending",
        "intent": normalized_intent,
        "expires_at": expires_at,
        "profile": None,
        "tokens": None,
        "claimed": False,
    }
    return pair, expires_at


def _extract_state_payload_unsafe(state_token: str) -> Optional[dict]:
    """Best-effort state payload read without signature validation (for UX recovery only)."""
    try:
        payload_b64 = state_token.split(".", 1)[0]
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _issue_oauth_state(payload: dict) -> str:
    """Create an opaque OAuth state token backed by server memory."""
    state_token = secrets.token_urlsafe(32)
    oauth_states[state_token] = {
        "payload": payload,
        "expires_at": _utc_now() + timedelta(seconds=PAIR_TTL_SECONDS),
    }
    return state_token


def _consume_oauth_state(state_token: str) -> Optional[dict]:
    """One-time state consumption for callback validation."""
    entry = oauth_states.pop(state_token, None)
    if not entry:
        return None

    expires_at = entry.get("expires_at")
    if expires_at and expires_at < _utc_now():
        return None

    payload = entry.get("payload")
    return payload if isinstance(payload, dict) else None


class PairCreateRequest(BaseModel):
    intent: str = "register"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _cleanup_expired() -> None:
    now = _utc_now()
    dead = [token for token, entry in pair_sessions.items() if entry.get("expires_at") and entry["expires_at"] < now]
    for token in dead:
        pair_sessions.pop(token, None)

    dead_states = [
        token
        for token, entry in oauth_states.items()
        if entry.get("expires_at") and entry["expires_at"] < now
    ]
    for token in dead_states:
        oauth_states.pop(token, None)


def _resolve_public_base(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL

    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host")
    scheme = forwarded_proto or request.url.scheme
    host = forwarded_host or request.url.netloc
    return f"{scheme}://{host}"


def _redirect_uri(request: Request) -> str:
    # Check module-level GOOGLE_OAUTH_REDIRECT_URI first (set from OAUTH_METHOD)
    if GOOGLE_OAUTH_REDIRECT_URI:
        return GOOGLE_OAUTH_REDIRECT_URI
    
    # Fallback to env var if not set via OAUTH_METHOD
    forced = os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "").strip()
    if forced:
        return forced
    return f"{_resolve_public_base(request)}/auth/google/callback"


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/pair/create")
async def pair_create(body: PairCreateRequest, request: Request):
    _cleanup_expired()

    intent = body.intent if body.intent in ("register", "login") else "register"
    pair, expires_at = _create_pair_session(intent=intent)

    base = _resolve_public_base(request)
    mobile_url = f"{base}/mobile-connect?pair={pair}"
    return {
        "pair": pair,
        "mobile_url": mobile_url,
        "expires_at": _iso(expires_at),
    }


@app.get("/mobile-connect", response_class=HTMLResponse)
async def mobile_connect(request: Request, pair: str = "", error: str = ""):
    _cleanup_expired()
    entry = pair_sessions.get(pair)
    if not pair or not entry:
        return HTMLResponse("<h3 style='font-family:sans-serif;padding:40px'>QR code expired. Please refresh on your main device and scan again.</h3>")

    intent = entry.get("intent", "register")
    retry_note = ""
    if error:
        retry_note = "<p style='background:#1e293b;border:1px solid #334155;padding:10px;border-radius:8px;margin:12px 0;color:#cbd5e1;'>Your previous sign-in session expired. Please continue with Google again.</p>"
    html = f"""
<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Continue with Google</title>
<style>
body {{ font-family: Arial, sans-serif; padding: 28px; background: #0f172a; color: #e2e8f0; }}
.card {{ max-width: 420px; margin: 40px auto; background: #111827; padding: 24px; border-radius: 12px; border: 1px solid #334155; }}
.btn {{ display: block; width: 100%; text-align: center; text-decoration: none; background: #2563eb; color: white; padding: 12px; border-radius: 8px; font-weight: 600; }}
.small {{ margin-top: 12px; color: #94a3b8; font-size: 13px; line-height: 1.6; }}
</style>
</head>
<body>
  <div class='card'>
    <h2 style='margin-top:0'>Google sign-in</h2>
        {retry_note}
    <p>Continue Google sign-in from your phone.</p>
    <a class='btn' href='/auth/google/start?pair={pair}&intent={intent}'>Continue with Google</a>
    <p class='small'>After success, go back to your mirror device and continue face setup there.</p>
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/auth/google/start")
async def auth_google_start(request: Request, pair: str = "", intent: str = "register"):
    _cleanup_expired()
    entry = pair_sessions.get(pair)
    if not entry:
        raise HTTPException(status_code=404, detail="Pair token expired")

    if entry.get("status") == "complete":
        return RedirectResponse(url="/pair-complete", status_code=302)

    if intent not in ("register", "login"):
        intent = entry.get("intent", "register")

    redirect_uri = _redirect_uri(request)
    code_verifier = secrets.token_urlsafe(64)
    state_payload = {
        "pair": pair,
        "intent": intent,
        "redirect_uri": redirect_uri,
        "cv": code_verifier,
        "ts": int(time.time()),
    }
    # Use signed state as primary so callback validation survives process restarts.
    # Callback also supports opaque in-memory state for backward compatibility.
    state = _sign_state(state_payload)

    try:
        auth_url, _ = build_auth_url_with_verifier(
            state=state,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
        )
    except FileNotFoundError as exc:
        msg = str(exc)
        html = f"""
<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>OAuth setup required</title>
<style>
body {{ font-family: Arial, sans-serif; padding: 28px; background: #0f172a; color: #e2e8f0; }}
.card {{ max-width: 560px; margin: 40px auto; background: #111827; padding: 24px; border-radius: 12px; border: 1px solid #334155; }}
.muted {{ color: #94a3b8; font-size: 14px; line-height: 1.65; }}
pre {{ background: #0b1220; border: 1px solid #334155; border-radius: 8px; padding: 12px; overflow: auto; color: #cbd5e1; }}
</style>
</head>
<body>
    <div class='card'>
        <h2 style='margin-top:0'>Google OAuth not configured on VPS</h2>
        <p class='muted'>The broker could not find OAuth client credentials, so sign-in cannot start yet.</p>
        <pre>{msg}</pre>
        <p class='muted'>Fix on VPS: place credentials_web.json in the project root, or set GOOGLE_OAUTH_CREDENTIALS_FILE to the full file path and restart the broker.</p>
    </div>
</body>
</html>
"""
        return HTMLResponse(html, status_code=500)
    except Exception as exc:
        return JSONResponse({"error": f"oauth_start_failed: {exc}"}, status_code=500)
    return RedirectResponse(url=auth_url)


@app.get("/auth/google/callback")
async def auth_google_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    _cleanup_expired()

    if error:
        pair, _ = _create_pair_session("register")
        return RedirectResponse(url=f"/mobile-connect?pair={pair}&error=oauth_error", status_code=302)

    if not code or not state:
        return JSONResponse({"error": "missing_code_or_state"}, status_code=400)

    # Prefer opaque one-time state tokens, with signed-state fallback.
    state_data = _consume_oauth_state(state or "")
    if not state_data:
        state_data = _decode_state(state or "")
    if not state_data:
        # If the state is correctly signed but expired, transparently restart OAuth.
        stale_state = _decode_state_allow_expired(state or "")
        if stale_state:
            pair = str(stale_state.get("pair", "") or "").strip()
            intent = str(stale_state.get("intent", "register") or "register").strip()
            if pair:
                _ensure_pair_entry(pair, intent=intent)
                return RedirectResponse(
                    url=f"/auth/google/start?pair={pair}&intent={intent}",
                    status_code=302,
                )

        unsafe_state = _extract_state_payload_unsafe(state or "") or {}
        recovered_pair = str(unsafe_state.get("pair", "") or "").strip()
        recovered_intent = str(unsafe_state.get("intent", "register") or "register").strip()

        if recovered_pair and recovered_pair in pair_sessions:
            return RedirectResponse(
                url=f"/mobile-connect?pair={recovered_pair}&error=state_reset",
                status_code=302,
            )

        fresh_pair, _ = _create_pair_session(recovered_intent)
        return RedirectResponse(
            url=f"/mobile-connect?pair={fresh_pair}&error=state_reset",
            status_code=302,
        )

    pair = state_data.get("pair", "")
    if not pair:
        return JSONResponse({"error": "pair_expired"}, status_code=400)

    intent = str(state_data.get("intent", "register") or "register").strip()
    entry = _ensure_pair_entry(pair, intent=intent)

    try:
        tokens, profile = exchange_code_for_tokens(
            code,
            redirect_uri=state_data.get("redirect_uri") or _redirect_uri(request),
            code_verifier=state_data.get("cv"),
        )
    except Exception as exc:
        return JSONResponse({"error": f"token_exchange_failed: {exc}"}, status_code=400)

    entry["tokens"] = tokens
    entry["profile"] = profile
    entry["status"] = "complete"

    html = """
<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Success</title>
<style>
body { font-family: Arial, sans-serif; padding: 28px; background: #0f172a; color: #e2e8f0; }
.card { max-width: 420px; margin: 40px auto; background: #111827; padding: 24px; border-radius: 12px; border: 1px solid #334155; text-align: center; }
</style>
</head>
<body>
  <div class='card'>
    <h2 style='margin-top:0'>Done</h2>
    <p>Google sign-in completed.</p>
    <p>You can return to your mirror device now.</p>
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/pair/status/{pair}")
async def pair_status(pair: str):
    _cleanup_expired()
    entry = pair_sessions.get(pair)
    if not entry:
        return {"status": "expired"}
    return {
        "status": entry.get("status", "pending"),
        "intent": entry.get("intent", "register"),
        "expires_at": _iso(entry["expires_at"]),
    }


@app.post("/pair/claim/{pair}")
async def pair_claim(pair: str):
    _cleanup_expired()
    entry = pair_sessions.get(pair)
    if not entry:
        raise HTTPException(status_code=404, detail="pair_expired")

    if entry.get("status") != "complete":
        raise HTTPException(status_code=409, detail="pair_not_ready")

    if entry.get("claimed"):
        raise HTTPException(status_code=409, detail="pair_already_claimed")

    profile = entry.get("profile")
    tokens = entry.get("tokens")
    if not profile or not tokens:
        raise HTTPException(status_code=500, detail="claim_payload_missing")

    entry["claimed"] = True
    pair_sessions.pop(pair, None)

    return {
        "profile": profile,
        "tokens": tokens,
    }


@app.get("/pair-complete", response_class=HTMLResponse)
async def pair_complete():
    return HTMLResponse("<h3 style='font-family:Arial,sans-serif;padding:40px'>Sign-in complete. You can return to your mirror device.</h3>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
