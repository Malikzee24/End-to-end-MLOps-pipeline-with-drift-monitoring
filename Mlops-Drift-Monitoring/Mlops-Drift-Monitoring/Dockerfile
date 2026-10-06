FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLOPS_STATE_DIR=/app/state \
    MLOPS_CONFIG=/app/configs/config.yaml

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY configs ./configs
COPY scripts ./scripts

RUN useradd -m appuser && mkdir -p /app/state && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["mlops", "serve", "--port", "8000"]
