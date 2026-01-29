FROM python:3.11-slim

WORKDIR /app

# System deps (curl + tar to fetch and unpack EZKL release artifacts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tar file \
  && rm -rf /var/lib/apt/lists/*

# ---- Install EZKL CLI (pin to 23.0.3) ----
# EZKL's Linux release artifact is a tarball named:
#   build-artifacts.ezkl-linux-gnu.tar.gz
# which contains an `ezkl` binary inside.
ARG EZKL_VERSION=23.0.3

RUN set -eux; \
  API_URL="https://api.github.com/repos/zkonduit/ezkl/releases/tags/v${EZKL_VERSION}"; \
  JSON="$(curl -fsSL "$API_URL")"; \
  FILE_URL="$(echo "$JSON" | grep -o 'https://github.com[^"]*' | grep 'build-artifacts.ezkl-linux-gnu.tar.gz' | head -n 1)"; \
  test -n "$FILE_URL"; \
  echo "Downloading: $FILE_URL"; \
  curl -fsSL -o /tmp/ezkl.tar.gz "$FILE_URL"; \
  mkdir -p /tmp/ezkl_unpack; \
  tar -xzf /tmp/ezkl.tar.gz -C /tmp/ezkl_unpack; \
  rm /tmp/ezkl.tar.gz; \
  # Find the ezkl binary in the extracted folder:
  EZKL_BIN_PATH="$(find /tmp/ezkl_unpack -type f -name ezkl | head -n 1)"; \
  test -n "$EZKL_BIN_PATH"; \
  cp "$EZKL_BIN_PATH" /usr/local/bin/ezkl; \
  chmod +x /usr/local/bin/ezkl; \
  file /usr/local/bin/ezkl | grep -E "ELF 64-bit"; \
  ezkl --version; \
  rm -rf /tmp/ezkl_unpack

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code + artifacts that ARE in repo
COPY app.py /app/app.py
COPY ezkl_artifacts /app/ezkl_artifacts

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
