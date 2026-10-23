# FlipRate — Production Deployment & Architecture Guide

A practical guide for systems engineers and architects integrating FlipRate into production infrastructure.

---

## 1. When to Use FlipRate (and When NOT to)

### Excellent Fit:
- **Customer Support Automation**: Gating incoming support tickets, chat inquiries, and feedback written in Hinglish, Romanized Indic scripts, or Devanagari.
- **Pre-LLM Sanitization**: Cleaning and standardizing prompt inputs to prevent prompt drift and hallucinations caused by noisy phonetic spelling.
- **Offline Model Audits**: Pre-release CI/CD regression tests measuring if a newly trained model is more fragile than the previous checkpoint.
- **Single-Process & Microservice APIs**: Running as a lightweight sidecar or in-process guardrail with zero external database dependencies.

### Not Recommended / Caution:
- **Non-Supported Native Scripts**: Scripts like Tamil, Telugu, Kannada, or Bengali in their native Brahmic alphabets are detected as `"other"` and routed to `REVIEW`. (Romanized transliterations of these languages are fully supported).
- **High-Throughput Ultra-Low Latency (<5ms) Hot Paths**: Generating multiple safe perturbations and running inference across them adds computational overhead. Use `fr.protect()` (pure normalization) or cached gating for latency-critical paths.
- **Unverified Custom Languages**: While FlipRate provides rules for 12+ Indian languages from curated datasets, domain-specific vocabularies should be reviewed on your real data before serving high-stakes traffic.

---

## 2. Architecture: Single-Instance vs Distributed Deployments

### A. Single-Instance Architecture (Default)
In single-process deployments (Docker container on Render, Railway, Hugging Face Spaces, or single VM):
- Zero external dependencies: no Redis, PostgreSQL, or message broker required.
- In-memory rate limiting and task tracking (`JobStore`) work out-of-the-box.
- Ideal for prototypes, internal tools, and medium-traffic APIs.

### B. Distributed / Multi-Worker Deployments (Behind a Load Balancer)
When scaling horizontally across multiple Kubernetes pods or behind an AWS ALB / NGINX load balancer:

> **Important Operational Note:**
> In-memory rate limits (`RateLimiter`) and background job trackers (`JobStore`) are process-local. If client requests hit different pods, rate limits will not be shared globally.

**Recommended Production Adjustments:**
1. **Stateless Runtime Gating**: Use `fr.quick_check()` or `fr.runtime_guard()` directly in your application server (e.g. FastAPI / Flask / Celery). The core gating algorithm is 100% stateless and thread-safe.
2. **API Gateway Rate Limiting**: Enforce global rate limits at your API gateway (Kong, Cloudflare, AWS API Gateway) rather than inside the Python application.
3. **Distributed Job Queue**: For large offline evaluation jobs (`fr.test` on 100K+ rows), offload batch processing to Celery, Temporal, or an SQS-backed worker pool rather than the local FastAPI background tasks.

---

## 3. Security & Configuration Best Practices

| Environment Variable | Production Recommendation | Rationale |
|---|---|---|
| `FLIPRATE_API_KEY` | Set a strong random secret (`openssl rand -hex 32`) | Protects public endpoints from unauthorized usage. |
| `FLIPRATE_CORS` | Set explicit allowed origin (e.g., `https://app.yourdomain.com`) | Never use `*` in production web environments. |
| `FLIPRATE_RATE_LIMIT` | Set per-client limits (e.g., `60`-`120` req/min) | Prevents DoS on inference endpoints. |
| `FLIPRATE_MAX_CONCURRENT`| `2`-`4` depending on CPU cores | Prevents memory exhaustion during heavy bootstrap test runs. |
| `FLIPRATE_MAX_TEXT_CHARS` | `1000`-`2000` | Rejects oversized payloads before tokenization. |

---

## 4. Data Privacy, Governance & PII Redaction

FlipRate processes conversational text which may contain sensitive personal data.

### Built-in Masking Protections:
- `fr.normalize()` automatically detects and protects URLs, email addresses, phone numbers, and @handles using internal masks, restoring them byte-for-byte.

### Enterprise Governance Checklist:
1. **No Data Retention by Default**: Core library functions do not write inputs to disk or external stores unless you explicitly call `report.save()`.
2. **Sanitize Logs**: When logging `GateResult` objects, avoid printing raw customer identifiers. Use `result.decision` and hashed text IDs.
3. **PII Scrubbing**: In highly regulated sectors (fintech, healthcare), run a PII scrubbing layer (such as Microsoft Presidio or regex anonymizers) before passing text to quality gates.

---

## 5. Observability & Monitoring

- **Health Probe**: `GET /health` returns `200 OK` for container readiness.
- **Model Readiness Probe**: `GET /ready` returns `503` while models or heavy pipelines are downloading/warming up, and `200` when ready.
- **Prometheus Metrics**: `GET /metrics` exports request counts, flip rate statistics, and latency histograms.
