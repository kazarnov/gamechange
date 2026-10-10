# The voice agent: the FastAPI app, the agent page and its models, on http://<host>:8000.
#   GPU (NVIDIA, default):  docker build -t flowai-agent .
#   CPU only (smaller):     docker build -t flowai-agent:cpu --build-arg TORCH=cpu .
#   With website tools:     add --build-arg BROWSER=true (installs Chromium for Playwright)
# Models are not baked in: they download on the first start into /cache (mount a volume there).
FROM python:3.12-slim-bookworm

# PyTorch build: cu128 runs on GTX 16xx up to RTX 50xx; cpu is about 3 GB smaller
ARG TORCH=cu128
ARG BROWSER=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/cache/huggingface \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    BROWSER_ENABLED=${BROWSER} \
    BROWSER_HEADLESS=true

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /usr/local/bin/uv

# PyTorch first and pinned, so requirements.txt can't swap it for another build
RUN uv pip install --system torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/${TORCH} \
 && printf 'torch==2.8.0\ntorchaudio==2.8.0\n' > /tmp/torch.txt
COPY requirements.txt /tmp/
RUN uv pip install --system -r /tmp/requirements.txt --constraint /tmp/torch.txt
# Compiled once here: starts faster, and old packages' syntax warnings stay out of the logs
RUN python -m compileall -q -j 0 /usr/local/lib/python3.12/site-packages > /dev/null 2>&1 || true
RUN if [ "$BROWSER" = "true" ]; then playwright install --with-deps chromium; fi

RUN useradd --create-home --uid 1000 agent \
 && mkdir -p /cache /app/media /app/voices /app/data \
 && chown agent /cache /app/media /app/voices /app/data
WORKDIR /app
COPY backend backend
COPY web web
COPY skills skills
COPY fonts fonts
COPY comfyui/descriptions.yaml comfyui/
USER agent

EXPOSE 8000
# The first start downloads the speech models (a few GB), so give it time
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
