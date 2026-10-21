FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Build with a real model:  docker build --build-arg EXTRAS=api,hf -t fliprate .
ARG EXTRAS=api
RUN pip install --no-cache-dir ".[${EXTRAS}]" && useradd -m -u 1000 app
USER app
# 7860 is the Hugging Face Spaces default; Render and others set $PORT themselves.
ENV PORT=7860 HF_HOME=/home/app/.cache/huggingface
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "uvicorn fliprate.api:app --host 0.0.0.0 --port ${PORT}"]
