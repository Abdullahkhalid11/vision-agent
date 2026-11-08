"""Tests for app/rate_limit.py with a fake clock. Run: python -m pytest tests/test_rate_limit.py -v"""
import time

from app.rate_limit import TokenBucketLimiter


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_21st_request_within_a_minute_is_denied():
    clock = FakeClock()
    limiter = TokenBucketLimiter(clock=clock)
    results = [limiter.allow("key-a") for _ in range(21)]
    assert results[:20] == [True] * 20
    assert results[20] is False


def test_one_request_succeeds_after_3_seconds():
    clock = FakeClock()
    limiter = TokenBucketLimiter(clock=clock)
    for _ in range(20):
        limiter.allow("key-a")
    assert limiter.allow("key-a") is False

    clock.advance(3)  # 20 tokens per 60 s = 1 token per 3 s
    assert limiter.allow("key-a") is True
    assert limiter.allow("key-a") is False  # only one token came back


def test_refill_never_exceeds_capacity():
    clock = FakeClock()
    limiter = TokenBucketLimiter(clock=clock)
    clock.advance(10_000)  # a long idle period
    results = [limiter.allow("key-a") for _ in range(21)]
    assert results[:20] == [True] * 20
    assert results[20] is False


def test_keys_have_independent_buckets():
    clock = FakeClock()
    limiter = TokenBucketLimiter(clock=clock)
    for _ in range(20):
        limiter.allow("key-a")
    assert limiter.allow("key-a") is False
    assert limiter.allow("key-b") is True


def test_default_clock_is_monotonic():
    assert TokenBucketLimiter().clock is time.monotonic
