# ============================================================================
# Dockerfile - Autonomous Multi-Asset Trading Agent (24/7 Cloud Ready)
# Optimiert für minimale RAM-Nutzung (< 350 MB) auf kostenlosen Free-Tier Servern
# ============================================================================
FROM python:3.10-slim

# Verhindert Pufferung von stdout/stderr (Echtzeit-Logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Zeitzone auf Europa/Berlin setzen
ENV TZ=Europe/Berlin

# Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungs-Code kopieren
COPY . .

# Daten-Verzeichnisse anlegen
RUN mkdir -p /app/data /app/data/synthesized_tools /app/data/synthesized_strategies /app/data/sec_form4_cache

# Standard-Befehl: 24/7 Dauer-Loop im 5-Minuten-Takt (300s)
CMD ["python", "main.py", "--loop", "300"]
