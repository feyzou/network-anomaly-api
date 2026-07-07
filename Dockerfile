# ── Stage 1 : builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2 : runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copier les dépendances depuis le builder
COPY --from=builder /install /usr/local

# Copier le code applicatif
COPY app/ ./app/
COPY model/ ./model/

# Permissions
RUN chown -R appuser:appuser /app
USER appuser

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/model/anomaly_detector.joblib \
    MODEL_VERSION=1.0.0

EXPOSE 8000

# Healthcheck Docker natif
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
