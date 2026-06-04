# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY thermalos/ ./thermalos/

RUN pip install --upgrade pip --quiet \
 && pip install build --quiet \
 && python -m build --wheel --outdir /dist

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="ThermalOS"
LABEL org.opencontainers.image.description="GPU thermal-power forensics agent"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/Asomisetty27/thermalos"

# Non-root user
RUN useradd --create-home --shell /bin/bash thermalos

WORKDIR /app
COPY --from=builder /dist/*.whl .
RUN pip install --quiet *.whl && rm *.whl

# Config and log dirs (writable by thermalos user)
RUN mkdir -p /home/thermalos/.thermalos /var/log/thermalos \
 && chown -R thermalos:thermalos /home/thermalos/.thermalos /var/log/thermalos

USER thermalos

# Prometheus metrics
EXPOSE 9101

# Defaults — override via env vars or command args
ENV THERMALOS_INTERVAL=5 \
    THERMALOS_PROMETHEUS_PORT=9101 \
    THERMALOS_LOG=/var/log/thermalos/alerts.jsonl

ENTRYPOINT ["thermalos"]
CMD ["monitor", \
     "--interval", "5", \
     "--port",     "9101", \
     "--log",      "/var/log/thermalos/alerts.jsonl"]
