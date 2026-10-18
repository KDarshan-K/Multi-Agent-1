"""
Lightweight TTL Caching for Tool Calls.

Prevents exhausting strict external API rate limits (such as Alpha Vantage's
5 calls/minute and 25 calls/day free tier) across revision rounds and runs.
"""

from __future__ import annotations
import time
from typing import Any, Callable

_CACHE: dict[str, tuple[float, Any]] = {}


def get_cached(key: str, ttl_seconds: float) -> tuple[bool, Any]:
    """Retrieve a cached item if present and not expired."""
    if key in _CACHE:
        timestamp, value = _CACHE[key]
        if time.time() - timestamp <= ttl_seconds:
            return True, value
        else:
            del _CACHE[key]
    return False, None


def set_cached(key: str, value: Any) -> None:
    """Store an item in the cache with the current timestamp."""
    _CACHE[key] = (time.time(), value)


def clear_cache() -> None:
    """Clear all cached entries."""
    _CACHE.clear()


def cached_call(cache_key: str, ttl_seconds: float, fn: Callable[[], Any]) -> Any:
    """Execute `fn` if cache key is missing or expired, otherwise return cached result."""
    hit, val = get_cached(cache_key, ttl_seconds)
    if hit:
        return val
    result = fn()
    set_cached(cache_key, result)
    return result
