FROM python:3.12-slim

WORKDIR /app

# Copy only what's needed for the install first (layer caching)
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Build with real HF model:  docker build --build-arg EXTRAS=api,hf -t fliprate .
ARG EXTRAS=api

# Install CPU-only torch when hf extra is requested to keep image size reasonable.
# The default torch wheel includes CUDA (~2 GB); torch-cpu is ~200 MB.
RUN if echo "${EXTRAS}" | grep -q "hf"; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    pip install --no-cache-dir ".[${EXTRAS}]"

# Non-root user for security
RUN useradd -m -u 1000 app
USER app

# 7860 is the Hugging Face Spaces default; Render/Railway/Fly set $PORT themselves.
ENV PORT=7860 \
    HF_HOME=/home/app/.cache/huggingface \
    FLIPRATE_API_KEY="" \
    FLIPRATE_CORS="*" \
    FLIPRATE_RATE_LIMIT="60" \
    FLIPRATE_MAX_CONCURRENT="2"

# Longer start period for XLM-R download on first boot
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health', timeout=4)"

# --proxy-headers trusts X-Forwarded-For from the platform's proxy.
# --forwarded-allow-ips="*" is safe only because the platform controls the network.
CMD ["sh", "-c", "uvicorn fliprate.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
