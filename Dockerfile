# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps (curl for pk.key download)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install EZKL (pin to the exact Colab version) and verify via python -m ezkl
RUN pip install --no-cache-dir ezkl==23.0.3 && python -m ezkl --version

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY app.py /app/app.py

# Copy all artifacts except pk.key (keep pk as release asset)
# Your repo should include: model.ezkl, vk.key, settings.json, model.onnx, kzg17.srs
COPY ezkl_artifacts /app/ezkl_artifacts

# Overwrite/ensure pk.key exists by downloading from GitHub Release
RUN mkdir -p /app/ezkl_artifacts && \
    curl -L -o /app/ezkl_artifacts/pk.key \
      https://github.com/Outerlands-capital/decision-integrity-proof-server/releases/download/artifacts-v1/pk.key

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
