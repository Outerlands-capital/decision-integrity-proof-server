# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# --- System deps ---
# curl + ca-certificates for downloads, tar/gzip for EZKL installer, file for sanity checks (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tar gzip file \
  && rm -rf /var/lib/apt/lists/*

# --- Install EZKL CLI (pinned) using official installer script ---
# The script downloads the correct build-artifacts tarball for the platform/arch.
# We install into /opt/ezkl and symlink to /usr/local/bin/ezkl.
ARG EZKL_TAG=v23.0.3
ENV EZKL_DIR=/opt/ezkl
RUN set -eux; \
    mkdir -p "${EZKL_DIR}"; \
    curl -fsSL https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash -s -- "${EZKL_TAG}"; \
    # The installer puts the binary into $EZKL_DIR/ezkl
    test -x "${EZKL_DIR}/ezkl"; \
    ln -sf "${EZKL_DIR}/ezkl" /usr/local/bin/ezkl; \
    ezkl --version

# --- Python deps ---
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- App code ---
COPY app.py /app/app.py

# --- EZKL artifacts folder (we'll populate it by downloading from GitHub Release) ---
RUN mkdir -p /app/ezkl_artifacts

# Provide URLs at build time (recommended) OR runtime (fallback handled in app.py)
# Example:
#   --build-arg MODEL_ONNX_URL=...
#   --build-arg SETTINGS_URL=...
#   --build-arg COMPILED_URL=...
#   --build-arg PK_URL=...
#   --build-arg VK_URL=...
#   --build-arg SRS_URL=...
ARG MODEL_ONNX_URL=""
ARG SETTINGS_URL=""
ARG COMPILED_URL=""
ARG PK_URL=""
ARG VK_URL=""
ARG SRS_URL=""

# Download artifacts if URLs were provided at build time
RUN set -eux; \
    dl () { \
      url="$1"; out="$2"; \
      if [ -n "$url" ]; then \
        echo "Downloading $url -> $out"; \
        curl -fL "$url" -o "$out"; \
      else \
        echo "Skipping $out (no URL provided)"; \
      fi; \
    }; \
    dl "${MODEL_ONNX_URL}" /app/ezkl_artifacts/model.onnx; \
    dl "${SETTINGS_URL}"   /app/ezkl_artifacts/settings.json; \
    dl "${COMPILED_URL}"   /app/ezkl_artifacts/model.ezkl; \
    dl "${PK_URL}"         /app/ezkl_artifacts/pk.key; \
    dl "${VK_URL}"         /app/ezkl_artifacts/vk.key; \
    dl "${SRS_URL}"        /app/ezkl_artifacts/kzg17.srs; \
    true

# Render uses PORT; expose for local
EXPOSE 8080
ENV PORT=8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
