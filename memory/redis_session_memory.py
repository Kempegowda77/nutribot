import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from memory.session_memory import SessionMemory

try:
    import redis
except Exception:  # pragma: no c over - handled at runtime
    redis = None


class RedisSessionStore:
    """
    Redis-backed session memory store with graceful fallback when redis-py is missing.
    """

    def __init__(
        self,
        redis_url: str,
        max_entries: int = 50,
        ttl_seconds: int = 86400,
    ) -> None:
        if redis is None:
            raise ImportError("redis package is not installed.")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.client = redis.from_url(redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"nutrichat:session:{session_id}"

    def load(self, session_id: str) -> SessionMemory:
        memory = SessionMemory(max_entries=self.max_entries)
        raw = self.client.get(self._key(session_id))
        if not raw:
            return memory

        try:
            items: List[Dict[str, object]] = json.loads(raw)
        except json.JSONDecodeError:
            return memory

        for item in items[-self.max_entries :]:
            ts = datetime.now(timezone.utc)
            timestamp = item.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    ts = datetime.fromisoformat(timestamp)
                except ValueError:
                    pass
            memory.add_entry(
                text=str(item.get("text", "")),
                value=float(item.get("value", 0.0)),
                timestamp=ts,
            )
        return memory

    def save(self, session_id: str, memory: SessionMemory) -> None:
        payload = json.dumps(memory.to_list(), ensure_ascii=True)
        key = self._key(session_id)
        self.client.set(key, payload, ex=self.ttl_seconds)

