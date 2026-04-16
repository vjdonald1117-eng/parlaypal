from fastapi import APIRouter

from api import _cache_store

router = APIRouter()


@router.get("/api/health")
async def health():
    latest = None
    for entry in _cache_store.values():
        ts = entry.get("timestamp")
        if isinstance(ts, str) and (latest is None or ts > latest):
            latest = ts
    return {"status": "ok", "cache_timestamp": latest}
