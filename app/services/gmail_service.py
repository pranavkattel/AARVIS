import os
import pickle
from pathlib import Path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
CONFIG_DIR = PROJECT_ROOT / 'config'
GMAIL_TOKEN_FILE = DATA_DIR / 'token_gmail.pickle'
CREDENTIALS_FILE = CONFIG_DIR / 'credentials.json'

# ── Per-user context ────────────────────────────────────────────────────────
_current_username: str | None = None

def set_current_user(username: str | None) -> None:
    """Set the active user so Gmail calls use their personal credentials."""
    global _current_username
    _current_username = username


def get_gmail_service(username: str | None = None):
    """
    Authenticate and return a Gmail service.

    If *username* is provided (or the module-level _current_username is set)
    the function loads that user's OAuth tokens that were stored during
    Google-sign-in registration.  Otherwise it falls back to the legacy
    single-user pickle-file approach.
    """
    resolved_user = username or _current_username

    # ── Per-user path (Google Sign-In users) ────────────────────────────────
    if resolved_user:
        try:
            from app.database import get_user_google_tokens, update_google_tokens
            from app.services.google_oauth import credentials_from_db
            token_row = get_user_google_tokens(resolved_user)
            if token_row:
                creds = credentials_from_db(token_row)
                # Persist any refreshed tokens back to the DB
                if creds.token != token_row.get("google_access_token"):
                    from datetime import timezone
                    update_google_tokens(resolved_user, {
                        "access_token":  creds.token,
                        "refresh_token": creds.refresh_token,
                        "expiry":        creds.expiry.isoformat() if creds.expiry else None,
                    })
                return build('gmail', 'v1', credentials=creds, cache_discovery=False)
        except Exception as e:
            print(f"[Gmail] Per-user credential load failed for {resolved_user}: {e}. Falling back to pickle.")

    # ── Legacy single-user path (pickle file) ───────────────────────────────
    creds = None

    if os.path.exists(GMAIL_TOKEN_FILE):
        with open(GMAIL_TOKEN_FILE, 'rb') as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(GMAIL_TOKEN_FILE, 'wb') as f:
            pickle.dump(creds, f)

    return build('gmail', 'v1', credentials=creds, cache_discovery=False)
