"""In-process login rate limiter (single uvicorn worker is the v1 target)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    failures: int = 0
    first_failure_at: float = 0.0
    locked_until: float = 0.0


@dataclass
class LoginRateLimiter:
    """Track failed logins per key (username or client host).

    After ``max_failures`` within ``window_seconds``, further attempts are
    rejected for ``lockout_seconds``.
    """

    max_failures: int = 5
    window_seconds: float = 300.0
    lockout_seconds: float = 300.0
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self, key: str) -> None:
        """Raise :class:`AuthError`-like ValueError when *key* is locked out.

        Callers map the message to ``AuthError("rate_limited", …)``.
        """
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return
            if bucket.locked_until > now:
                remaining = int(bucket.locked_until - now) + 1
                raise ValueError(f"Too many failed login attempts; try again in {remaining}s")
            # Window expired → reset quietly.
            if (
                bucket.failures
                and now - bucket.first_failure_at > self.window_seconds
                and bucket.locked_until <= now
            ):
                self._buckets.pop(key, None)

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or now - bucket.first_failure_at > self.window_seconds:
                bucket = _Bucket(failures=0, first_failure_at=now)
                self._buckets[key] = bucket
            bucket.failures += 1
            if bucket.failures >= self.max_failures:
                bucket.locked_until = now + self.lockout_seconds

    def record_success(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)


# Process-global limiter (one per serve process).
default_login_limiter = LoginRateLimiter()
