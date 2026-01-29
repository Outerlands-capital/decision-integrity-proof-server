FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install the ezkl wheel (contains the native binary inside), then extract the binary to /usr/local/bin/ezkl
RUN pip install --no-cache-dir ezkl==23.0.3 && \
    python - <<'PY'
import os, stat, shutil
from pathlib import Path
import ezkl

pkg_dir = Path(ezkl.__file__).resolve().parent
candidates = []

# Find files literally named "ezkl" under the package directory
for p in pkg_dir.rglob("ezkl"):
    if p.is_file():
        # skip python sources
        if p.suffix in (".py", ".pyc"):
            continue
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        candidates.append((size, p))

if not candidates:
    raise SystemExit(f"Could not find embedded ezkl binary inside package dir: {pkg_dir}")

# pick the largest "ezkl" file (usually the native binary)
candidates.sort(reverse=True, key=lambda x: x[0])
bin_path = candidates[0][1]

dst = Path("/usr/local/bin/ezkl")
shutil.copy2(bin_path, dst)
dst.chmod(dst.stat().st_mode | stat.S_IEXEC)

print("Embedded ezkl binary found at:", bin_path)
print("Installed ezkl binary to:", dst)
PY

# Validate ezkl works
RUN /usr/local/bin/ezkl --version

# Python deps for your server
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY app.py /app/app.py

# Artifacts (repo contains all but pk.key)
COPY ezkl_artifacts /app/ezkl_artifacts

# Download pk.key from GitHub Release (big file)
RUN mkdir -p /app/ezkl_artifacts && \
    curl -L -o /app/ezkl_artifacts/pk.key \
      https://github.com/Outerlands-capital/decision-integrity-proof-server/releases/download/artifacts-v1/pk.key

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
