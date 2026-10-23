"""FastAPI service: start a job with POST, watch it live with Server-Sent Events.

Run:   uvicorn fliprate.api:app
Model: FLIPRATE_MODEL=my_package.my_module:my_function   (default: toy keyword model)
Env:   FLIPRATE_API_KEY, FLIPRATE_CORS, FLIPRATE_RATE_LIMIT, FLIPRATE_MAX_CONCURRENT, FLIPRATE_JOB_TTL, FLIPRATE_REDIS_URL
"""
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .demo import load_model, sample_dataset
from .events import to_sse
from .service import (
    Metrics,
    Settings,
    create_job_store,
    create_rate_limiter,
    key_ok,
    run_job_events,
)

log = logging.getLogger("fliprate")
SETTINGS = Settings.from_env()
MODEL = load_model()
JOBS = create_job_store(SETTINGS)
LIMITER = create_rate_limiter(SETTINGS)
METRICS = Metrics()
SEM = threading.BoundedSemaphore(SETTINGS.max_concurrent_runs)
READY = {"model": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SETTINGS.env == "production" and "*" in SETTINGS.cors_origins:
        log.warning(
            "SECURITY WARNING: FLIPRATE_CORS is set to '*' in production. "
            "Please restrict FLIPRATE_CORS to your authorized frontend domains."
        )
    try:  # load the model before the first request so users do not wait on a cold start
        MODEL("warmup")
        READY["model"] = True
    except Exception:
        log.exception("model warmup failed")
    yield


app = FastAPI(title="FlipRate API", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Assign or forward an X-Request-ID correlation header and log structured metrics."""
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    t0 = time.time()
    response: Response = await call_next(request)
    duration_ms = round((time.time() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = req_id
    log.info(
        "request_completed method=%s path=%s status=%d duration_ms=%.2f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        req_id,
    )
    return response


class AnalyzeIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=SETTINGS.max_text_chars)
    n_variants: int = Field(8, ge=1, le=30)


class Row(BaseModel):
    text: str
    label: str


class TestIn(BaseModel):
    n_variants: int = Field(5, ge=1, le=20)
    rows: Optional[List[Row]] = None  # omit to use the bundled sample


def _dump(m: BaseModel) -> Dict:
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()  # pydantic v2 / v1


def _guard(request: Request, x_api_key: Optional[str]) -> None:
    if not key_ok(x_api_key, SETTINGS.api_key):
        METRICS.inc("auth_failures_total")
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
    who = x_api_key or (request.client.host if request.client else "unknown")
    if not LIMITER.allow(who):
        METRICS.inc("rate_limited_total")
        raise HTTPException(status_code=429, detail="Too many requests")


@app.post("/jobs/analyze")
def start_analyze(body: AnalyzeIn, request: Request, x_api_key: Optional[str] = Header(None)):
    """Text goes in the POST body, never in a URL."""
    _guard(request, x_api_key)
    METRICS.inc("jobs_started_total")
    return {"job_id": JOBS.put("analyze", _dump(body))}


@app.post("/jobs/test")
def start_test(body: TestIn, request: Request, x_api_key: Optional[str] = Header(None)):
    _guard(request, x_api_key)
    if body.rows is not None and len(body.rows) > SETTINGS.max_rows:
        raise HTTPException(status_code=422, detail=f"At most {SETTINGS.max_rows} rows per job")
    METRICS.inc("jobs_started_total")
    return {"job_id": JOBS.put("test", _dump(body))}


@app.get("/jobs/{job_id}/stream")
def stream(job_id: str):
    """The job id is a one-time secret token, so this endpoint needs no header (EventSource cannot send one)."""
    job = JOBS.pop(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown, expired or already-streamed job")

    def body():
        for ev in run_job_events(job, MODEL, sample_dataset, SEM, METRICS):
            yield to_sse(ev)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready():
    if not READY["model"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return METRICS.render()


_WEB = Path(__file__).parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
def console():
    return FileResponse(
        _WEB,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
