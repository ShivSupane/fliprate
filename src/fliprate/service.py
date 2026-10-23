"""Pure-Python building blocks for the web service (testable without FastAPI)."""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Generator, List, Optional

from .evaluation import iter_test
from .events import Event, make_event
from .gating import iter_gate

log = logging.getLogger("fliprate")


@dataclass
class Settings:
    api_key: Optional[str] = None          # if set, POST endpoints require header X-API-Key
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_min: int = 60           # per API key or client IP; 0 disables
    max_jobs: int = 500
    job_ttl_seconds: int = 300
    max_concurrent_runs: int = 2
    max_text_chars: int = 2000
    max_rows: int = 2000
    redis_url: Optional[str] = None        # if set, uses Redis for distributed jobs & rate limiting
    env: str = "development"               # "production" or "development"

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        e = os.environ if env is None else env
        return cls(
            api_key=e.get("FLIPRATE_API_KEY") or None,
            cors_origins=[o.strip() for o in e.get("FLIPRATE_CORS", "*").split(",") if o.strip()],
            rate_limit_per_min=int(e.get("FLIPRATE_RATE_LIMIT", 60)),
            max_jobs=int(e.get("FLIPRATE_MAX_JOBS", 500)),
            job_ttl_seconds=int(e.get("FLIPRATE_JOB_TTL", 300)),
            max_concurrent_runs=int(e.get("FLIPRATE_MAX_CONCURRENT", 2)),
            max_text_chars=int(e.get("FLIPRATE_MAX_TEXT_CHARS", 2000)),
            max_rows=int(e.get("FLIPRATE_MAX_ROWS", 2000)),
            redis_url=e.get("FLIPRATE_REDIS_URL") or None,
            env=e.get("FLIPRATE_ENV", "development").lower(),
        )


def key_ok(supplied: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time API key check. No key configured means open access (fine for local use only)."""
    if not expected:
        return True
    return supplied is not None and hmac.compare_digest(supplied.encode(), expected.encode())


class BaseJobStore(ABC):
    """Abstract interface for background job storage."""

    @abstractmethod
    def put(self, kind: str, params: Dict[str, Any]) -> str:
        """Store job parameters and return a unique unguessable job ID."""
        pass

    @abstractmethod
    def pop(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and remove job parameters by job ID."""
        pass


class JobStore(BaseJobStore):
    """In-memory one-time job storage (single process)."""

    def __init__(self, max_jobs: int = 500, ttl: float = 300, clock: Callable[[], float] = time.monotonic):
        self.max_jobs, self.ttl, self.clock = max_jobs, ttl, clock
        self._jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, kind: str, params: Dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {"kind": kind, "params": params, "created": self.clock()}
            while len(self._jobs) > self.max_jobs:
                self._jobs.popitem(last=False)
        return job_id

    def pop(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None or self.clock() - job["created"] > self.ttl:
            return None
        return job

    def __len__(self) -> int:
        return len(self._jobs)


InMemoryJobStore = JobStore


class RedisJobStore(BaseJobStore):
    """Distributed Redis-backed job storage for multi-worker environments."""

    def __init__(self, redis_url: str, ttl: int = 300, key_prefix: str = "fliprate:job:"):
        try:
            import redis
            self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        except ImportError as exc:
            raise ImportError("RedisJobStore requires `pip install redis`") from exc
        self.ttl = ttl
        self.key_prefix = key_prefix

    def put(self, kind: str, params: Dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        key = f"{self.key_prefix}{job_id}"
        data = json.dumps({"kind": kind, "params": params, "created": time.time()})
        self.client.setex(key, self.ttl, data)
        return job_id

    def pop(self, job_id: str) -> Optional[Dict[str, Any]]:
        key = f"{self.key_prefix}{job_id}"
        val = self.client.get(key)
        if val is None:
            return None
        self.client.delete(key)
        return json.loads(val)


class BaseRateLimiter(ABC):
    """Abstract interface for rate limiting."""

    @abstractmethod
    def allow(self, key: str) -> bool:
        """Return True if request is allowed, False if limit exceeded."""
        pass


class RateLimiter(BaseRateLimiter):
    """Sliding-window in-memory rate limiter per key (single process)."""

    def __init__(self, limit_per_min: int, clock: Callable[[], float] = time.monotonic):
        self.limit, self.clock = limit_per_min, clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._last_seen: Dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = self.clock()
        with self._lock:
            stale = [k for k, t in self._last_seen.items() if now - t > 120]
            for k in stale:
                self._hits.pop(k, None)
                self._last_seen.pop(k, None)
            self._last_seen[key] = now
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


InMemoryRateLimiter = RateLimiter


class RedisRateLimiter(BaseRateLimiter):
    """Distributed Redis rate limiter using fixed/sliding window counters."""

    def __init__(self, redis_url: str, limit_per_min: int, key_prefix: str = "fliprate:rl:"):
        try:
            import redis
            self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        except ImportError as exc:
            raise ImportError("RedisRateLimiter requires `pip install redis`") from exc
        self.limit = limit_per_min
        self.key_prefix = key_prefix

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        current_minute = int(time.time() // 60)
        redis_key = f"{self.key_prefix}{key}:{current_minute}"
        try:
            pipe = self.client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, 120)
            count, _ = pipe.execute()
            return count <= self.limit
        except Exception as exc:
            log.warning("Redis rate limiter connection error: %s, allowing request", exc)
            return True


def create_job_store(settings: Settings) -> BaseJobStore:
    """Factory creating appropriate JobStore based on configuration."""
    if settings.redis_url:
        try:
            return RedisJobStore(settings.redis_url, ttl=settings.job_ttl_seconds)
        except Exception as exc:
            log.warning("Could not connect to Redis for JobStore: %s. Falling back to in-memory.", exc)
    return InMemoryJobStore(settings.max_jobs, settings.job_ttl_seconds)


def create_rate_limiter(settings: Settings) -> BaseRateLimiter:
    """Factory creating appropriate RateLimiter based on configuration."""
    if settings.redis_url:
        try:
            return RedisRateLimiter(settings.redis_url, limit_per_min=settings.rate_limit_per_min)
        except Exception as exc:
            log.warning("Could not connect to Redis for RateLimiter: %s. Falling back to in-memory.", exc)
    return InMemoryRateLimiter(settings.rate_limit_per_min)


class Metrics:
    """Tiny counters rendered in Prometheus text format at /metrics. Never records message text."""

    def __init__(self):
        self._c: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._c[name] += value

    def render(self) -> str:
        with self._lock:
            return "".join(f"fliprate_{k} {v}\n" for k, v in sorted(self._c.items()))


def run_job_events(job: Dict[str, Any], model: Callable, sample_rows: Callable[[], list],
                   sem: threading.BoundedSemaphore, metrics: Metrics) -> Generator[Event, None, None]:
    """Yield the job's events. Errors are logged (type only, never text) and reported generically."""
    p = job["params"]
    if not sem.acquire(blocking=False):
        metrics.inc("jobs_rejected_busy_total")
        yield make_event("error", message="Server is busy, try again shortly")
        yield make_event("end")
        return
    t0 = time.time()
    try:
        if job["kind"] == "analyze":
            yield from iter_gate(model, p["text"], n_variants=p["n_variants"])
        else:
            rows = [(r["text"], r["label"]) for r in p["rows"]] if p.get("rows") else sample_rows()
            yield from iter_test(model, rows, n_variants=p["n_variants"])
        metrics.inc(f"jobs_{job['kind']}_ok_total")
    except Exception as exc:
        metrics.inc("jobs_failed_total")
        log.error("job failed kind=%s error=%s", job["kind"], type(exc).__name__)
        log.debug("job failure details", exc_info=True)
        yield make_event("error", message=f"The model failed ({type(exc).__name__}). Check the server logs.")
    finally:
        sem.release()
        metrics.inc("job_seconds_total", time.time() - t0)
    yield make_event("end")
