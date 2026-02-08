"""
Google OAuth helper for per-user Gmail / Calendar access.

Flow
----
1.  build_auth_url()           → redirect the user to Google
2.  exchange_code_for_tokens() → called in the /auth/google/callback handler
                                 returns (credentials_dict, profile_dict)
3.  credentials_from_db()      → rebuild a Credentials object from what is
                                 stored in the database so the Gmail / Calendar
                                 services can use it directly.

Google Cloud Console requirements
----------------------------------
- Create an OAuth 2.0 **Web application** credential.
- Add  http://localhost:8000/auth/google/callback  as an authorized redirect URI.
- Download it as  credentials_web.json  into the config folder.
  (The existing credentials.json is an "installed" / desktop type and cannot
   be used for server-side redirect flows.)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow

# ── Config ──────────────────────────────────────────────────────────────────

# Preferred: web-app credential file.  Falls back to the installed-app file so
# the server still starts while the user upgrades their Google Cloud project.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
_WEB_CREDS_FILE = CONFIG_DIR / "credentials_web.json"
_INSTALLED_CREDS_FILE = CONFIG_DIR / "credentials.json"

REDIRECT_URI = "http://localhost:8000/auth/google/callback"

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]


def _get_client_config() -> dict:
    """Load the OAuth client configuration from disk."""
    # 1) Explicit override for VPS/container deployments.
    forced_path = os.getenv("GOOGLE_OAUTH_CREDENTIALS_FILE", "").strip()
    if forced_path:
        forced = Path(forced_path)
        if forced.exists():
            with open(forced) as f:
                return json.load(f)

    # 2) Common local/deployment locations.
    candidate_web_files = [
        _WEB_CREDS_FILE,
        Path.cwd() / "config" / "credentials_web.json",
        Path.cwd() / "config" / "credentials.json",
    ]
    for web_file in candidate_web_files:
        if web_file.exists():
            with open(web_file) as f:
                raw = json.load(f)
            if "web" in raw:
                return raw
            if "installed" in raw:
                installed = raw.get("installed", {})
                return {
                    "web": {
                        "client_id": installed["client_id"],
                        "client_secret": installed["client_secret"],
                        "auth_uri": installed.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
                        "token_uri": installed.get("token_uri", "https://oauth2.googleapis.com/token"),
                        "redirect_uris": [REDIRECT_URI],
                    }
                }

    # Fallback – convert the installed-app JSON into a pseudo-web format so
    # Flow.from_client_config() accepts it.
    if _INSTALLED_CREDS_FILE.exists():
        with open(_INSTALLED_CREDS_FILE) as f:
            raw = json.load(f)
        # Promote "installed" → "web" so the library handles it the same way
        installed = raw.get("installed", {})
        return {
            "web": {
                "client_id": installed["client_id"],
                "client_secret": installed["client_secret"],
                "auth_uri": installed.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": installed.get("token_uri", "https://oauth2.googleapis.com/token"),
                "redirect_uris": [REDIRECT_URI],
            }
        }

    raise FileNotFoundError(
        "No Google OAuth credentials file found. "
        "Place credentials_web.json (Web application type) in the config folder "
        "or set GOOGLE_OAUTH_CREDENTIALS_FILE to its full path."
    )


def build_auth_url(state: str = "", redirect_uri: str = None) -> str:
    """Return the Google OAuth consent-screen URL to redirect the user to."""
    uri = redirect_uri or REDIRECT_URI
    config = _get_client_config()
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=uri)
    auth_url, _ = flow.authorization_url(
        access_type="offline",      # get a refresh_token
        include_granted_scopes="true",
        prompt="select_account",    # show account picker only; skip consent for returning users
        state=state,
    )
    return auth_url


def build_auth_url_with_verifier(
    state: str = "",
    redirect_uri: str = None,
    code_verifier: str | None = None,
) -> tuple[str, str | None]:
    """Return consent URL and PKCE verifier (if generated by the OAuth library)."""
    uri = redirect_uri or REDIRECT_URI
    config = _get_client_config()
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=uri)
    if code_verifier:
        flow.code_verifier = code_verifier
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="select_account",
        state=state,
    )
    return auth_url, getattr(flow, "code_verifier", None)


def exchange_code_for_tokens(code: str, redirect_uri: str = None, code_verifier: str | None = None) -> tuple[dict, dict]:
    """
    Exchange an authorization code for tokens and fetch the user's profile.

    Returns
    -------
    tokens : dict   – keys: access_token, refresh_token, token_uri, client_id,
                             client_secret, scopes, expiry (ISO string or None)
    profile : dict  – keys: google_id, email, full_name, picture
    """
    uri = redirect_uri or REDIRECT_URI
    config = _get_client_config()
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=uri)
    if code_verifier:
        flow.code_verifier = code_verifier
    flow.fetch_token(code=code)
    creds = flow.credentials

    # ── fetch profile via Google's userinfo endpoint ──────────────────────
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {creds.token}"},
        )
        with urllib.request.urlopen(req) as resp:
            profile_raw = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to fetch Google profile: {exc}") from exc

    profile = {
        "google_id": profile_raw.get("sub"),
        "email": profile_raw.get("email"),
        "full_name": profile_raw.get("name"),
        "picture": profile_raw.get("picture"),
    }

    expiry_str = creds.expiry.isoformat() if creds.expiry else None

    tokens = {
        "access_token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or SCOPES),
        "expiry": expiry_str,
    }

    return tokens, profile


def credentials_from_db(token_row: dict) -> Credentials:
    """
    Rebuild a google.oauth2.credentials.Credentials object from a DB row.

    The row must have: access_token, refresh_token, token_uri, client_id,
                       client_secret, google_scopes (comma-separated), google_token_expiry
    Automatically refreshes the credential if it's expired.
    """
    expiry = None
    if token_row.get("google_token_expiry"):
        try:
            expiry = datetime.fromisoformat(token_row["google_token_expiry"])
            # google-auth compares expiry using datetime.utcnow() (naive),
            # so we must keep expiry naive UTC — strip tzinfo if present.
            if expiry.tzinfo is not None:
                expiry = expiry.replace(tzinfo=None)
        except ValueError:
            pass

    scopes_raw = token_row.get("google_scopes") or ",".join(SCOPES)
    scopes = [s.strip() for s in scopes_raw.split(",") if s.strip()]

    creds = Credentials(
        token=token_row["google_access_token"],
        refresh_token=token_row.get("google_refresh_token"),
        token_uri=token_row.get("google_token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_row.get("google_client_id"),
        client_secret=token_row.get("google_client_secret"),
        scopes=scopes,
        expiry=expiry,
    )

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return creds
