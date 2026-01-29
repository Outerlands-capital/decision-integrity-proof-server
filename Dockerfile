FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install EZKL CLI via pip (pins your exact version)
RUN pip install --no-cache-dir ezkl==23.0.3 && ezkl --version

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

# artifacts
RUN mkdir -p /app/ezkl_artifacts
# (either COPY the folder or download from your GitHub release)
# COPY ezkl_artifacts /app/ezkl_artifacts
RUN curl -L -o /app/ezkl_artifacts/pk.key \
  https://github.com/Outerlands-capital/decision-integrity-proof-server/releases/download/artifacts-v1/pk.key

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
