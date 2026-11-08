"""Token bucket rate limiter. One bucket per key. Thread-safe. The clock is injected."""
import threading
import time
from dataclasses import dataclass

CAPACITY = 20          # maximum tokens a bucket holds, and the burst size
WINDOW_SECONDS = 60.0  # the bucket refills CAPACITY tokens over this many seconds


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, capacity=CAPACITY, window_seconds=WINDOW_SECONDS, clock=time.monotonic):
        self.capacity = capacity
        self.window_seconds = window_seconds
        self.clock = clock
        self._buckets = {}
        self._lock = threading.Lock()

    def allow(self, key):
        """Return True and spend one token if the key has one; otherwise return False."""
        with self._lock:
            now = self.clock()
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.capacity), updated=now)
                self._buckets[key] = bucket

            # Refill for the time since the last call, never above capacity.
            elapsed = now - bucket.updated
            refilled = elapsed * self.capacity / self.window_seconds
            bucket.tokens = min(float(self.capacity), bucket.tokens + refilled)
            bucket.updated = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False
