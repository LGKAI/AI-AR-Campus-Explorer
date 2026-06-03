# engine/storage.py
"""
SQLite-backed persistent storage cho User Profiles.
====================================================
Thay thế in-memory USER_PROFILES_DB: dict = {} trong api/main.py.

Vấn đề giải quyết:
  - Profile mất khi server restart (Critical bug)
  - Session không có TTL → RAM tăng vô hạn
  - Không thể chia sẻ state giữa nhiều worker process

Thiết kế:
  - WAL mode: đọc/ghi đồng thời an toàn (phù hợp FastAPI async)
  - RAM cache: giữ hot session trong memory để tránh I/O mỗi request
  - TTL 48h: tự dọn session cũ
"""

import json
import hashlib
import hmac
import os
import sqlite3
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

_DB_PATH = Path(__file__).parent / "campus_users.db"
_SALT_PATH = Path(__file__).parent / ".privacy_salt"
_SESSION_TTL_HOURS: int = 48       # Session hết hạn sau 48h không hoạt động
_CACHE_MAX_SIZE: int = 500         # RAM cache tối đa 500 session hot

# Thread-safe RAM cache: session_id → (profile, last_access_ts)
_ram_cache: Dict[str, dict] = {}
_cache_lock = Lock()


# ---------------------------------------------------------------------------
# Khởi tạo DB
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Tạo bảng nếu chưa có. Gọi một lần khi server start."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                session_id   TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at   REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiles_updated
            ON user_profiles(updated_at)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id      TEXT NOT NULL,
                session_id   TEXT NOT NULL,
                content      TEXT NOT NULL,
                created_at   REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_comments_node
            ON comments(node_id)
        """)
        # WAL mode: concurrent reads + writes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    print(f"✅ [Storage] SQLite DB sẵn sàng: {_DB_PATH}")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def load_profile(session_id: str) -> Optional[dict]:
    """
    Đọc profile theo thứ tự ưu tiên:
      1. RAM cache (nếu còn trong TTL)
      2. SQLite DB
    Trả về None nếu không tồn tại hoặc đã expired.
    """
    with _cache_lock:
        if session_id in _ram_cache:
            return _ram_cache[session_id]

    cutoff = time.time() - _SESSION_TTL_HOURS * 3600
    with _connect() as conn:
        row = conn.execute(
            "SELECT profile_json FROM user_profiles WHERE session_id=? AND updated_at>?",
            (session_id, cutoff),
        ).fetchone()

    if row is None:
        return None

    profile = json.loads(row["profile_json"])
    with _cache_lock:
        _evict_cache_if_needed()
        _ram_cache[session_id] = profile
    return profile


def save_profile(session_id: str, profile: dict) -> None:
    """Ghi profile vào RAM cache và SQLite."""
    now = time.time()
    profile["_anon_user_id"] = anonymize_session_id(session_id)
    with _cache_lock:
        _evict_cache_if_needed()
        _ram_cache[session_id] = profile

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (session_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at   = excluded.updated_at
            """,
            (session_id, json.dumps(profile, ensure_ascii=False, default=str), now),
        )


def delete_expired_sessions() -> int:
    """
    Xóa session quá TTL khỏi DB và RAM cache.
    Gọi định kỳ (ví dụ mỗi 1 giờ).
    Trả về số session đã xóa.
    """
    cutoff = time.time() - _SESSION_TTL_HOURS * 3600
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM user_profiles WHERE updated_at < ?", (cutoff,)
        )
        deleted = cur.rowcount

    # Dọn RAM cache theo danh sách đã xóa trong DB
    with _cache_lock:
        live_ids = _get_live_session_ids()
        stale = [sid for sid in list(_ram_cache) if sid not in live_ids]
        for sid in stale:
            _ram_cache.pop(sid, None)

    if deleted:
        print(f"🧹 [Storage] Đã dọn {deleted} session hết hạn.")
    return deleted


def get_all_profiles() -> Dict[str, dict]:
    """
    Trả về tất cả profile đang active với khóa người dùng đã ẩn danh.
    Model chỉ thấy anon_user_id, không thấy session_id gốc.
    """
    cutoff = time.time() - _SESSION_TTL_HOURS * 3600
    result: Dict[str, dict] = {}

    with _connect() as conn:
        rows = conn.execute(
            "SELECT session_id, profile_json FROM user_profiles WHERE updated_at > ?",
            (cutoff,),
        ).fetchall()

    for row in rows:
        profile = json.loads(row["profile_json"])
        anon_id = profile.get("_anon_user_id") or anonymize_session_id(row["session_id"])
        profile["_anon_user_id"] = anon_id
        result[anon_id] = _training_profile(profile)

    # Merge RAM cache (có thể mới hơn DB)
    with _cache_lock:
        for sid, profile in _ram_cache.items():
            anon_id = profile.get("_anon_user_id") or anonymize_session_id(sid)
            profile["_anon_user_id"] = anon_id
            result[anon_id] = _training_profile(profile)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evict_cache_if_needed() -> None:
    """LRU-lite: xóa bớt khi cache đầy (giữ _CACHE_MAX_SIZE mới nhất)."""
    if len(_ram_cache) >= _CACHE_MAX_SIZE:
        # Đơn giản: xóa 10% cũ nhất (không cần LRU hoàn chỉnh)
        n_evict = max(1, _CACHE_MAX_SIZE // 10)
        for sid in list(_ram_cache.keys())[:n_evict]:
            _ram_cache.pop(sid, None)


def _get_live_session_ids() -> set:
    cutoff = time.time() - _SESSION_TTL_HOURS * 3600
    with _connect() as conn:
        rows = conn.execute(
            "SELECT session_id FROM user_profiles WHERE updated_at > ?", (cutoff,)
        ).fetchall()
    return {r["session_id"] for r in rows}


def anonymize_session_id(session_id: str) -> str:
    """Tạo mã người dùng ổn định cho ML mà không lộ session_id gốc."""
    sid = session_id or "default"
    digest = hmac.new(_privacy_salt(), sid.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"u_{digest[:16]}"


def _privacy_salt() -> bytes:
    env_salt = os.environ.get("CAMPUS_PRIVACY_SALT")
    if env_salt:
        return env_salt.encode("utf-8")
    if not _SALT_PATH.exists():
        _SALT_PATH.write_text(os.urandom(32).hex(), encoding="utf-8")
    return _SALT_PATH.read_text(encoding="utf-8").strip().encode("utf-8")


def _training_profile(profile: dict) -> dict:
    """Bản profile cho training, bỏ các trường định danh/nội bộ không cần thiết."""
    return {k: v for k, v in profile.items() if k not in {"session_id", "_last_tracking"}}


# ---------------------------------------------------------------------------
# COMMENTS CRUD
# ---------------------------------------------------------------------------

def save_comment(node_id: str, session_id: str, content: str) -> None:
    """Lưu bình luận mới vào SQLite."""
    delete_old_comments()
    now = time.time()
    anon_id = anonymize_session_id(session_id)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO comments (node_id, session_id, content, created_at) VALUES (?, ?, ?, ?)",
            (node_id, anon_id, content, now),
        )


def get_comments(node_id: str) -> list:
    """Lấy bình luận trong ngày hiện tại của một địa điểm."""
    delete_old_comments()
    cutoff = _today_start_ts()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT session_id, content, created_at
            FROM comments
            WHERE node_id=? AND created_at>=?
            ORDER BY created_at DESC
            """,
            (node_id, cutoff),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_old_comments() -> int:
    """Xóa bình luận từ các ngày trước để feed chỉ sống trong ngày."""
    cutoff = _today_start_ts()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM comments WHERE created_at < ?", (cutoff,))
        return cur.rowcount


def _today_start_ts() -> float:
    today = datetime.now().date()
    return datetime.combine(today, dt_time.min).timestamp()
