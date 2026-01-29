FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates file \
  && rm -rf /var/lib/apt/lists/*

# ---- Install EZKL binary (v23.0.3) from GitHub releases ----
# Important: use -f (fail on HTTP error) and verify it's a Linux ELF binary.
ARG EZKL_VERSION=23.0.3
RUN set -eux; \
    curl -fL -o /usr/local/bin/ezkl \
      https://github.com/zkonduit/ezkl/releases/download/v${EZKL_VERSION}/ezkl-linux-amd64; \
    chmod +x /usr/local/bin/ezkl; \
    file /usr/local/bin/ezkl | grep -E "ELF 64-bit LSB executable"; \
    /usr/local/bin/ezkl --version

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App
COPY app.py /app/app.py

# Artifacts (repo contains all but pk.key)
COPY ezkl_artifacts /app/ezkl_artifacts

# Download pk.key from GitHub Release (big file)
RUN set -eux; \
    mkdir -p /app/ezkl_artifacts; \
    curl -fL -o /app/ezkl_artifacts/pk.key \
      https://github.com/Outerlands-capital/decision-integrity-proof-server/releases/download/artifacts-v1/pk.key; \
    test -s /app/ezkl_artifacts/pk.key

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
