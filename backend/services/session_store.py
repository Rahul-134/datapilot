"""
Per-session key/value storage for uploaded/cleaned/query/scrape data.

On a traditional long-running server, a module-level dict is enough because the
process stays alive between requests. On Vercel, each request can land on a
different (or freshly cold-started) function instance, so anything kept only in
process memory can vanish between an upload and the next request that needs it.

If UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN are set (e.g. via the
Upstash integration on Vercel), state is stored in Redis, keyed by a
per-browser session id supplied by the frontend, so it survives across
function instances. Locally, where those env vars are typically unset, it
falls back to an in-process dict so `uvicorn backend.main:app --reload`
keeps working without needing Redis installed.
"""

import os
import requests

_UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
_UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
_TTL_SECONDS = 7200  # sessions expire after 2 hours of inactivity

_local_store: dict = {}

# Every key any route stores per-session — used by clear_session() to wipe a session fully.
SESSION_KEYS = ["df", "filename", "cleaned_df", "cleaned_filename", "query_df", "query_prompt", "scrape_df"]


def _using_redis() -> bool:
    return bool(_UPSTASH_URL and _UPSTASH_TOKEN)


def _cmd(*args):
    resp = requests.post(
        _UPSTASH_URL,
        headers={"Authorization": f"Bearer {_UPSTASH_TOKEN}"},
        json=list(args),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("result")


def _full_key(session_id: str, key: str) -> str:
    return f"session:{session_id}:{key}"


def set_value(session_id: str, key: str, value: str):
    full_key = _full_key(session_id, key)
    if _using_redis():
        _cmd("SET", full_key, value, "EX", _TTL_SECONDS)
    else:
        _local_store[full_key] = value


def get_value(session_id: str, key: str):
    full_key = _full_key(session_id, key)
    if _using_redis():
        return _cmd("GET", full_key)
    return _local_store.get(full_key)


def delete_value(session_id: str, key: str):
    full_key = _full_key(session_id, key)
    if _using_redis():
        _cmd("DEL", full_key)
    else:
        _local_store.pop(full_key, None)


def clear_session(session_id: str):
    for key in SESSION_KEYS:
        delete_value(session_id, key)
