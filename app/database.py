import sqlite3
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "smart_mirror.db"

def get_db():
    """Get database connection with timeout to prevent locking"""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent access
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

def init_db():
    """Initialize database with users table"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            full_name TEXT NOT NULL,
            location TEXT NOT NULL DEFAULT '',
            interests TEXT,
            google_id TEXT UNIQUE,
            google_access_token TEXT,
            google_refresh_token TEXT,
            google_token_uri TEXT,
            google_client_id TEXT,
            google_client_secret TEXT,
            google_scopes TEXT,
            google_token_expiry TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Migrate: fix NOT NULL on password_hash / full_name / location ─────────
    col_info = {row[1]: row for row in cursor.execute("PRAGMA table_info(users)")}
    needs_null_fix = (
        col_info.get("password_hash") and col_info["password_hash"][3] == 1  # notnull
    )
    if needs_null_fix:
        # SQLite can't ALTER COLUMN, so we recreate the table
        existing_user_cols = [row[1] for row in cursor.execute("PRAGMA table_info(users)")]
        cursor.execute("PRAGMA foreign_keys=OFF")
        cursor.execute("ALTER TABLE users RENAME TO _users_old")
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                full_name TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                interests TEXT,
                google_id TEXT,
                google_access_token TEXT,
                google_refresh_token TEXT,
                google_token_uri TEXT,
                google_client_id TEXT,
                google_client_secret TEXT,
                google_scopes TEXT,
                google_token_expiry TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        shared_cols = [c for c in [
            'id','username','email','password_hash','full_name','location',
            'interests','created_at','google_id','google_access_token',
            'google_refresh_token','google_token_uri','google_client_id',
            'google_client_secret','google_scopes','google_token_expiry'
        ] if c in existing_user_cols]
        cols_str = ", ".join(shared_cols)
        cursor.execute(f"INSERT INTO users ({cols_str}) SELECT {cols_str} FROM _users_old")
        cursor.execute("DROP TABLE _users_old")
        cursor.execute("PRAGMA foreign_keys=ON")
        conn.commit()

    # ── Migrate existing databases: add Google OAuth columns if missing ──────
    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(users)")}
    google_cols = {
        "google_id":            "TEXT",
        "google_access_token":  "TEXT",
        "google_refresh_token": "TEXT",
        "google_token_uri":     "TEXT",
        "google_client_id":     "TEXT",
        "google_client_secret": "TEXT",
        "google_scopes":        "TEXT",
        "google_token_expiry":  "TEXT",
    }
    for col, col_type in google_cols.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
    # Create unique index for google_id separately (ALTER TABLE can't add UNIQUE columns)
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_id ON users (google_id)
        WHERE google_id IS NOT NULL
    """)
    
    # Face embeddings table for LVFace verification
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            embedding_version TEXT DEFAULT 'lvface_v1',
            face_photo_path TEXT,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    
    # Attendance records table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verification_score REAL,
            method TEXT DEFAULT 'face_verification',
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    
    # Conversation history table for LLM memory and context
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            intent TEXT,
            agent_type TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversation_user_session 
        ON conversation_history(user_id, session_id, created_at DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversation_user_time 
        ON conversation_history(user_id, created_at DESC)
    """)
    
    conn.commit()
    conn.close()
    print("[DEBUG] Database initialized successfully")

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_google_user(
    google_id: str,
    email: str,
    full_name: str,
    tokens: dict,
    location: str = "",
    interests: str = "",
) -> int:
    """
    Create (or update) a user that authenticated via Google OAuth.

    tokens dict keys expected:
        access_token, refresh_token, token_uri, client_id,
        client_secret, scopes (list), expiry (ISO str or None)

    Returns the user's database id.
    """
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()

        scopes_str = ",".join(tokens.get("scopes") or [])
        expiry_str = tokens.get("expiry") or None

        # Derive a safe username from the email local-part
        base_username = email.split("@")[0].replace(".", "_").replace("+", "_")
        # Make it unique if already taken
        username = base_username
        suffix = 1
        while True:
            row = cursor.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if row is None:
                break
            # Already exists — check if it belongs to this google_id
            owner = cursor.execute(
                "SELECT google_id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if owner and owner["google_id"] == google_id:
                break
            username = f"{base_username}_{suffix}"
            suffix += 1

        existing = cursor.execute(
            "SELECT id FROM users WHERE google_id = ?", (google_id,)
        ).fetchone()

        # Also check by email — handles the case where the user previously
        # registered with email/password and now signs in with Google.
        if not existing:
            existing = cursor.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()

        if existing:
            # User already exists — link Google account and refresh tokens
            cursor.execute(
                """
                UPDATE users SET
                    google_id            = ?,
                    google_access_token  = ?,
                    google_refresh_token = COALESCE(?, google_refresh_token),
                    google_token_uri     = ?,
                    google_client_id     = ?,
                    google_client_secret = ?,
                    google_scopes        = ?,
                    google_token_expiry  = ?,
                    full_name            = ?
                WHERE id = ?
                """,
                (
                    google_id,
                    tokens.get("access_token"),
                    tokens.get("refresh_token"),
                    tokens.get("token_uri"),
                    tokens.get("client_id"),
                    tokens.get("client_secret"),
                    scopes_str,
                    expiry_str,
                    full_name,
                    existing["id"],
                ),
            )
            conn.commit()
            return existing["id"], False   # (id, is_new)
        else:
            # Brand-new user — insert fresh record
            cursor.execute(
                """
                INSERT INTO users (
                    username, email, password_hash, full_name, location, interests,
                    google_id, google_access_token, google_refresh_token,
                    google_token_uri, google_client_id, google_client_secret,
                    google_scopes, google_token_expiry
                ) VALUES (?, ?, NULL, ?, ?, ?,
                          ?, ?, ?,
                          ?, ?, ?,
                          ?, ?)
                """,
                (
                    username, email, full_name, location, interests,
                    google_id,
                    tokens.get("access_token"),
                    tokens.get("refresh_token"),
                    tokens.get("token_uri"),
                    tokens.get("client_id"),
                    tokens.get("client_secret"),
                    scopes_str,
                    expiry_str,
                ),
            )
            conn.commit()
            row = cursor.execute("SELECT id FROM users WHERE google_id = ?", (google_id,)).fetchone()
            return row["id"], True   # (id, is_new)
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def update_google_tokens(username: str, tokens: dict) -> None:
    """Persist refreshed Google OAuth tokens for an existing user."""
    conn = None
    try:
        conn = get_db()
        scopes_str = ",".join(tokens.get("scopes") or [])
        conn.execute(
            """
            UPDATE users SET
                google_access_token  = ?,
                google_refresh_token = COALESCE(?, google_refresh_token),
                google_token_expiry  = ?
            WHERE username = ?
            """,
            (
                tokens.get("access_token"),
                tokens.get("refresh_token"),
                tokens.get("expiry"),
                username,
            ),
        )
        conn.commit()
    finally:
        if conn:
            conn.close()


def get_user_google_tokens(username: str) -> dict | None:
    """
    Return the Google OAuth token row for a user, or None if not a Google user.
    The returned dict contains all columns needed by services/google_oauth.credentials_from_db().
    """
    conn = None
    try:
        conn = get_db()
        row = conn.execute(
            """
            SELECT google_access_token, google_refresh_token, google_token_uri,
                   google_client_id, google_client_secret, google_scopes,
                   google_token_expiry
            FROM users WHERE username = ?
            """,
            (username,),
        ).fetchone()
        if row and row["google_access_token"]:
            return dict(row)
        return None
    finally:
        if conn:
            conn.close()


def get_user_by_google_id(google_id: str) -> dict | None:
    """Fetch a user record by their Google sub/ID."""
    conn = None
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT id, username, email, full_name, location, interests FROM users WHERE google_id = ?",
            (google_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        if conn:
            conn.close()


def create_user(username: str, email: str, password: str, full_name: str, location: str, interests: str = ""):
    """Create a new user"""
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, location, interests)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, email, password_hash, full_name, location, interests))
        
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError as e:
        if conn:
            conn.rollback()
        if "username" in str(e):
            raise ValueError("Username already exists")
        elif "email" in str(e):
            raise ValueError("Email already exists")
        raise ValueError("User creation failed")
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def verify_user(username: str, password: str):
    """Verify user credentials"""
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute("""
            SELECT id, username, email, full_name, location, interests
            FROM users
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))
        
        user = cursor.fetchone()
        
        if user:
            return dict(user)
        return None
    finally:
        if conn:
            conn.close()

def get_user_by_username(username: str):
    """Get user by username"""
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, email, full_name, location, interests, google_id
            FROM users
            WHERE username = ?
        """, (username,))
        
        user = cursor.fetchone()
        
        if user:
            return dict(user)
        return None
    finally:
        if conn:
            conn.close()


def get_all_users() -> list:
    """Return all users (admin use)."""
    conn = None
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT id, username, email, full_name, location, interests,
                   google_id, created_at
            FROM users ORDER BY id
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        if conn: conn.close()


def delete_user_by_id(user_id: int) -> bool:
    """Delete a user by id. Returns True if a row was removed."""
    conn = None
    try:
        conn = get_db()
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        if conn: conn.close()


def admin_update_user(user_id: int, full_name: str = None, email: str = None,
                      location: str = None, interests: str = None) -> bool:
    """Update editable fields for a user. Returns True if updated."""
    conn = None
    try:
        conn = get_db()
        fields, vals = [], []
        if full_name  is not None: fields.append("full_name = ?");  vals.append(full_name)
        if email      is not None: fields.append("email = ?");      vals.append(email)
        if location   is not None: fields.append("location = ?");   vals.append(location)
        if interests  is not None: fields.append("interests = ?");  vals.append(interests)
        if not fields: return False
        vals.append(user_id)
        cur = conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", vals)
        conn.commit()
        return cur.rowcount > 0
    finally:
        if conn: conn.close()

def update_user_preferences(username: str, location: str = None, interests: str = None):
    """Update user preferences"""
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        if location:
            cursor.execute("UPDATE users SET location = ? WHERE username = ?", (location, username))
        if interests:
            cursor.execute("UPDATE users SET interests = ? WHERE username = ?", (interests, username))
        
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


# ============================================================================
# Face Embedding Functions for LVFace
# ============================================================================

def save_face_embedding(user_id: int, embedding: np.ndarray, photo_path: str = None, version: str = 'lvface_v1'):
    """
    Save face embedding for a user
    Args:
        user_id: User ID from users table
        embedding: 512D numpy array from LVFace
        photo_path: Optional path to enrollment photo
        version: Model version identifier
    Returns:
        embedding_id or raises ValueError
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Convert embedding to bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        cursor.execute("""
            INSERT INTO face_embeddings (user_id, embedding, embedding_version, face_photo_path)
            VALUES (?, ?, ?, ?)
        """, (user_id, embedding_bytes, version, photo_path))
        
        conn.commit()
        embedding_id = cursor.lastrowid
        conn.close()
        return embedding_id
    except Exception as e:
        raise ValueError(f"Failed to save face embedding: {e}")


def get_face_embedding(user_id: int) -> np.ndarray:
    """
    Retrieve face embedding for a user
    Args:
        user_id: User ID
    Returns:
        512D numpy array or None if not found
    """
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT embedding FROM face_embeddings 
        WHERE user_id = ? 
        ORDER BY enrolled_at DESC 
        LIMIT 1
    """, (user_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        embedding_bytes = result[0]
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        return embedding
    return None


def get_all_face_embeddings() -> dict:
    """
    Get all face embeddings from database
    Returns:
        Dict of {user_id: embedding}
    """
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT user_id, embedding 
        FROM face_embeddings 
        ORDER BY enrolled_at DESC
    """)
    
    embeddings = {}
    for row in cursor.fetchall():
        user_id = row[0]
        if user_id not in embeddings:  # Only take most recent
            embedding_bytes = row[1]
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings[user_id] = embedding
    
    conn.close()
    return embeddings


def update_face_embedding(user_id: int, embedding: np.ndarray):
    """
    Update existing face embedding for a user
    Args:
        user_id: User ID
        embedding: New 512D embedding
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        cursor.execute("""
            UPDATE face_embeddings 
            SET embedding = ?, enrolled_at = CURRENT_TIMESTAMP 
            WHERE user_id = ?
        """, (embedding_bytes, user_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Failed to update face embedding: {e}")
        return False


def delete_face_embedding(user_id: int):
    """Delete face embedding for a user"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM face_embeddings WHERE user_id = ?", (user_id,))
    
    conn.commit()
    conn.close()


def has_face_embedding(user_id: int) -> bool:
    """Check if user has face embedding enrolled"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    
    return count > 0


# ============================================================================
# Attendance Functions
# ============================================================================

def mark_attendance(user_id: int, verification_score: float = None, method: str = 'face_verification'):
    """
    Mark attendance for a user
    Args:
        user_id: User ID
        verification_score: Cosine similarity score
        method: Verification method used
    Returns:
        attendance_id
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO attendance (user_id, verification_score, method)
            VALUES (?, ?, ?)
        """, (user_id, verification_score, method))
        
        conn.commit()
        attendance_id = cursor.lastrowid
        conn.close()
        return attendance_id
    except Exception as e:
        raise ValueError(f"Failed to mark attendance: {e}")


def get_attendance_today(user_id: int = None):
    """
    Get attendance records for today
    Args:
        user_id: Optional filter by user
    Returns:
        List of attendance records
    """
    conn = get_db()
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("""
            SELECT a.id, a.user_id, u.username, u.full_name, a.check_in_time, a.verification_score
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE a.user_id = ? AND DATE(a.check_in_time) = DATE('now')
            ORDER BY a.check_in_time DESC
        """, (user_id,))
    else:
        cursor.execute("""
            SELECT a.id, a.user_id, u.username, u.full_name, a.check_in_time, a.verification_score
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE DATE(a.check_in_time) = DATE('now')
            ORDER BY a.check_in_time DESC
        """)
    
    records = [dict(zip(['id', 'user_id', 'username', 'full_name', 'check_in_time', 'verification_score'], row)) 
               for row in cursor.fetchall()]
    
    conn.close()
    return records


def get_attendance_history(user_id: int, days: int = 30):
    """Get attendance history for a user"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, check_in_time, verification_score, method
        FROM attendance
        WHERE user_id = ? AND check_in_time >= datetime('now', '-' || ? || ' days')
        ORDER BY check_in_time DESC
    """, (user_id, days))
    
    records = [dict(zip(['id', 'check_in_time', 'verification_score', 'method'], row)) 
               for row in cursor.fetchall()]
    
    conn.close()
    return records


# ============================================================================
# Conversation History Functions for LLM Memory
# ============================================================================

def save_conversation(user_id: int, session_id: str, role: str, content: str, 
                     intent: str = None, agent_type: str = None, metadata: str = None):
    """
    Save a conversation message to history
    Args:
        user_id: User ID
        session_id: Session identifier (UUID or timestamp-based)
        role: 'user', 'assistant', or 'system'
        content: Message content
        intent: Detected intent (e.g., 'email', 'calendar', 'general')
        agent_type: Agent that processed (e.g., 'EmailAgent', 'GeneralAgent')
        metadata: JSON string with additional data (confidence, entities, etc.)
    Returns:
        conversation_id
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_history 
            (user_id, session_id, role, content, intent, agent_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, session_id, role, content, intent, agent_type, metadata))
        
        conn.commit()
        conversation_id = cursor.lastrowid
        conn.close()
        return conversation_id
    except Exception as e:
        raise ValueError(f"Failed to save conversation: {e}")


def get_conversation_history(user_id: int, session_id: str = None, limit: int = 50):
    """
    Get conversation history for a user
    Args:
        user_id: User ID
        session_id: Optional session filter
        limit: Maximum number of messages to retrieve
    Returns:
        List of conversation messages
    """
    conn = get_db()
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute("""
            SELECT id, session_id, role, content, intent, agent_type, metadata, created_at
            FROM conversation_history
            WHERE user_id = ? AND session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, session_id, limit))
    else:
        cursor.execute("""
            SELECT id, session_id, role, content, intent, agent_type, metadata, created_at
            FROM conversation_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
    
    records = [dict(zip(['id', 'session_id', 'role', 'content', 'intent', 'agent_type', 'metadata', 'created_at'], row)) 
               for row in cursor.fetchall()]
    
    conn.close()
    return records


def get_recent_context(user_id: int, limit: int = 10):
    """
    Get recent conversation context for building LLM prompts
    Args:
        user_id: User ID
        limit: Number of recent messages (default 10 for context window)
    Returns:
        List of recent messages in chronological order
    """
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT role, content, intent, created_at
        FROM conversation_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit))
    
    # Reverse to get chronological order (oldest first)
    records = [dict(zip(['role', 'content', 'intent', 'created_at'], row)) 
               for row in cursor.fetchall()]
    
    conn.close()
    return list(reversed(records))


def clear_old_conversations(days: int = 90):
    """
    Clear conversation history older than specified days
    Args:
        days: Keep conversations newer than this many days
    Returns:
        Number of deleted records
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM conversation_history
            WHERE created_at < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted_count
    except Exception as e:
        print(f"Failed to clear old conversations: {e}")
        return 0


def get_conversation_stats(user_id: int):
    """
    Get conversation statistics for a user
    Returns:
        Dictionary with stats
    """
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_messages,
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
            COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages,
            MAX(created_at) as last_interaction
        FROM conversation_history
        WHERE user_id = ?
    """, (user_id,))
    
    row = cursor.fetchone()
    stats = dict(zip(['total_messages', 'total_sessions', 'user_messages', 'assistant_messages', 'last_interaction'], row))
    
    # Get intent breakdown
    cursor.execute("""
        SELECT intent, COUNT(*) as count
        FROM conversation_history
        WHERE user_id = ? AND intent IS NOT NULL
        GROUP BY intent
        ORDER BY count DESC
    """, (user_id,))
    
    stats['intent_breakdown'] = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    return stats


# Initialize database on import
init_db()
