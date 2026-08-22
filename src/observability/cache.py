from collections import OrderedDict
from typing import Any, Optional, Dict
import hashlib
import time


class LRUCache:
    """
    Fast In-Memory LRU Cache with TTL (Time To Live) support for queries and embeddings.
    """

    def __init__(self, max_size: int = 1000, ttl_sec: float = 3600.0):
        self.max_size = max_size
        self.ttl_sec = ttl_sec
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.strip().lower().encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        k = self._hash_key(key)
        if k not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[k]
        if time.time() - entry["timestamp"] > self.ttl_sec:
            del self.cache[k]
            self.misses += 1
            return None

        self.cache.move_to_end(k)
        self.hits += 1
        return entry["value"]

    def put(self, key: str, value: Any) -> None:
        k = self._hash_key(key)
        if k in self.cache:
            self.cache.move_to_end(k)
        self.cache[k] = {
            "value": value,
            "timestamp": time.time()
        }
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round((self.hits / total) * 100, 2) if total > 0 else 0.0
