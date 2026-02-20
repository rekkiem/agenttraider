# ─────────────────────────────────────────────────────────────────────
# AgentTrader CL — Dockerfile
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── Dependencias del sistema ──────────────────────────────────────────
# Playwright necesita Chromium y sus dependencias
RUN apt-get update && apt-get install -y \
    # Chromium para Playwright
    chromium \
    chromium-driver \
    # Librerías para Prophet (opcional)
    libgomp1 \
    # Herramientas básicas
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Instalar navegadores de Playwright
RUN playwright install chromium \
    && playwright install-deps chromium

# ── Código fuente ─────────────────────────────────────────────────────
COPY . .

# ── Directorios de trabajo ────────────────────────────────────────────
RUN mkdir -p logs saved_models

# ── Variables de entorno para Playwright (usar Chromium del sistema) ──
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# ── Healthcheck ───────────────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD python -c "import psycopg2; psycopg2.connect(host='postgres', dbname='agenttrader', user='trader', password='${DB_PASSWORD}')" || exit 1

# ── Entry point ───────────────────────────────────────────────────────
CMD ["python", "main.py", "--mode", "scheduler"]
