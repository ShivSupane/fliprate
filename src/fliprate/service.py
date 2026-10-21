"""Pure-Python building blocks for the web service (testable without FastAPI)."""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
import uuid
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
        )


def key_ok(supplied: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time API key check. No key configured means open access (fine for local use only)."""
    if not expected:
        return True
    return supplied is not None and hmac.compare_digest(supplied.encode(), expected.encode())


class JobStore:
    """One-time jobs. The job id is an unguessable token, so the stream URL needs no header
    (the browser's EventSource cannot send custom headers)."""

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


class RateLimiter:
    """Sliding-window limiter, per key. In-memory, so per process: use a shared store if you scale out."""

    def __init__(self, limit_per_min: int, clock: Callable[[], float] = time.monotonic):
        self.limit, self.clock = limit_per_min, clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


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
        # Log the error TYPE only. Exception messages can contain user text, so the traceback is DEBUG-only.
        log.error("job failed kind=%s error=%s", job["kind"], type(exc).__name__)
        log.debug("job failure details", exc_info=True)
        yield make_event("error", message=f"The model failed ({type(exc).__name__}). Check the server logs.")
    finally:
        sem.release()
        metrics.inc("job_seconds_total", time.time() - t0)
    yield make_event("end")
