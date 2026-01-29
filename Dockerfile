FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install EZKL 23.0.3 (linux amd64)
# If the asset name differs, we’ll adjust to the exact release asset name.
RUN curl -L -o /usr/local/bin/ezkl \
    https://github.com/zkonduit/ezkl/releases/download/v23.0.3/ezkl-linux-amd64 \
  && chmod +x /usr/local/bin/ezkl \
  && ezkl --version

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY app.py /app/app.py

# Artifacts dir (your code expects /app/ezkl_artifacts)
RUN mkdir -p /app/ezkl_artifacts

# Download pk.key from your GitHub Release asset
RUN curl -L -o /app/ezkl_artifacts/pk.key \
  https://github.com/Outerlands-capital/decision-integrity-proof-server/releases/download/artifacts-v1/pk.key

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
