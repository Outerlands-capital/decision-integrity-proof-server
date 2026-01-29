# app.py
"""
Decision Integrity Proof Server (Real ZK version via EZKL)

- /prove generates a REAL zk proof using EZKL
- /verify verifies it using vk.key

Artifacts:
- model.onnx, settings.json, model.ezkl, vk.key are in repo under ./ezkl_artifacts
- pk.key is LARGE, so it is downloaded at runtime from EZKL_PK_URL if missing
- kzg17.srs is in repo under ./ezkl_artifacts (optional URL override via EZKL_SRS_URL)

Env:
  APP_ENV=dev
  ADMIN_KEY=...

  EZKL_BIN=ezkl
  EZKL_ARTIFACTS_DIR=./ezkl_artifacts

  # Optional override paths (defaults to ARTIFACTS_DIR/*)
  EZKL_MODEL_ONNX=...
  EZKL_SETTINGS=...
  EZKL_COMPILED=...
  EZKL_PK=...
  EZKL_VK=...
  EZKL_SRS=...

  # URLs (used if local file missing)
  EZKL_PK_URL=https://.../pk.key
  EZKL_SRS_URL=https://.../kzg17.srs   (optional)

Option 2 (server-side only, no API changes):
- Store commit records in-memory keyed by commitment_hash.
- On /reveal, enforce that the CURRENT model hash equals the model hash at commit time.
  If model hash was rotated after commit -> commitment_matches becomes false.

FIXES:
- Make verify deterministic on Render by using staged default filenames in the temp working dir
  (vk.key + model.compiled + settings.json + srs staged into cwd), and using relative paths.
- Log verify failures to Render logs without changing API schemas.
"""

import base64
import hashlib
import json
import math
import os
import subprocess
import tempfile
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock
import urllib.request

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic import ConfigDict

APP_ENV = os.getenv("APP_ENV", "dev")
ADMIN_KEY = os.getenv("ADMIN_KEY", "dev-admin-key")

# Demo toggle: when enabled, server will intentionally corrupt proofs to show verification failures
TAMPER_PROOF = False

DEFAULT_MODEL_HASH = "sha256:geo-escalation-7d-demo-v1"

MODELS = [
    {
        "model_id": "geo_escalation_7d_v1",
        "description": "Geopolitical escalation risk forecast (next 7 days) — Demo (Real ZK via EZKL)",
        "model_hash": DEFAULT_MODEL_HASH,
        "feature_schema": [
            "analyst_a_assessment",
            "analyst_b_assessment",
            "analyst_c_assessment",
            "analyst_d_assessment",
        ],
        "policy_note": "Demo uses abstracted public indicators / analyst judgments only. Not operational intelligence.",
    }
]

# =========================
# Option 2: In-memory commit store (server-side only)
# =========================
COMMIT_DB: Dict[str, Dict[str, Any]] = {}
COMMIT_DB_LOCK = Lock()
COMMIT_TTL_SEC = int(os.getenv("COMMIT_TTL_SEC", "3600"))  # demo TTL (default 1 hour)

def _prune_commit_db(now: Optional[float] = None):
    now = now or time.time()
    cutoff = now - COMMIT_TTL_SEC
    with COMMIT_DB_LOCK:
        to_delete = [k for k, v in COMMIT_DB.items() if float(v.get("created_at", 0)) < cutoff]
        for k in to_delete:
            del COMMIT_DB[k]

# -------------------------
# EZKL configuration
# -------------------------
EZKL_BIN = os.getenv("EZKL_BIN", "ezkl")
EZKL_ARTIFACTS_DIR = Path(os.getenv("EZKL_ARTIFACTS_DIR", "./ezkl_artifacts")).resolve()

EZKL_MODEL_ONNX = Path(os.getenv("EZKL_MODEL_ONNX", str(EZKL_ARTIFACTS_DIR / "model.onnx"))).resolve()
EZKL_SETTINGS   = Path(os.getenv("EZKL_SETTINGS",   str(EZKL_ARTIFACTS_DIR / "settings.json"))).resolve()
EZKL_COMPILED   = Path(os.getenv("EZKL_COMPILED",   str(EZKL_ARTIFACTS_DIR / "model.ezkl"))).resolve()
EZKL_PK         = Path(os.getenv("EZKL_PK",         str(EZKL_ARTIFACTS_DIR / "pk.key"))).resolve()
EZKL_VK         = Path(os.getenv("EZKL_VK",         str(EZKL_ARTIFACTS_DIR / "vk.key"))).resolve()
EZKL_SRS        = Path(os.getenv("EZKL_SRS",        str(EZKL_ARTIFACTS_DIR / "kzg17.srs"))).resolve()

EZKL_PK_URL = os.getenv("EZKL_PK_URL", "").strip()
EZKL_SRS_URL = os.getenv("EZKL_SRS_URL", "").strip()

STRICT_EZKL = os.getenv("STRICT_EZKL", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Decision Integrity Proof Server", version="0.4.2-real-zk-ezkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic base (avoid "model_" namespace warnings in Pydantic v2)
# =========================
class ApiModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# =========================
# Request/Response Models (FINAL - do not change)
# =========================
class ProveRequest(ApiModel):
    model_id: str = Field(..., description="Which approved model version to use")
    features: List[float] = Field(..., description="Numeric feature vector")
    context: Optional[Dict[str, Any]] = Field(default=None)


class ProveResponse(ApiModel):
    model_id: str
    model_hash: str
    prediction: float
    proof_b64: str
    public_inputs_b64: str


class VerifyRequest(ApiModel):
    model_id: str
    proof_b64: str
    public_inputs_b64: str


class VerifyResponse(ApiModel):
    model_id: str
    valid: bool


class CommitRequest(ApiModel):
    model_id: str
    prediction: float
    nonce: str
    context: Optional[Dict[str, Any]] = None


class CommitResponse(ApiModel):
    commitment_hash: str


class CommitCaseRequest(ApiModel):
    model_id: str
    features: List[float]
    nonce: str
    context: Optional[Dict[str, Any]] = None


class CommitCaseResponse(ApiModel):
    commitment_hash: str
    features_hash: str


class RevealRequest(ApiModel):
    model_id: str
    features: List[float]
    nonce: str
    context: Optional[Dict[str, Any]] = None
    expected_commitment_hash: str


class RevealResponse(ApiModel):
    model_id: str
    model_hash: str
    prediction: float
    proof_b64: str
    public_inputs_b64: str
    commitment_hash: str
    commitment_matches: bool
    features_hash: str


class AdminToggleRequest(ApiModel):
    enabled: bool


class AdminSetModelHashRequest(ApiModel):
    model_id: str
    model_hash: str


# =========================
# Helpers
# =========================
def get_model(model_id: str) -> Dict[str, Any]:
    model = next((m for m in MODELS if m["model_id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Unknown model_id")
    return model


def require_admin(x_admin_key: Optional[str]):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


def enforce_policy_constraints(features: List[float]):
    if len(features) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 features for this PoC.")
    if any((x < -10.0 or x > 10.0) for x in features[:4]):
        raise HTTPException(status_code=400, detail="Feature out of allowed range (-10..10).")


def maybe_tamper_proof(proof_b64: str) -> str:
    if not proof_b64:
        return proof_b64
    first = "A" if proof_b64[0] != "A" else "B"
    return first + proof_b64[1:]


def features_hash_hex(features: List[float]) -> str:
    payload = json.dumps(features, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def compute_commitment(model_hash: str, feat_hash_hex: str, nonce: str, context: Dict[str, Any]) -> str:
    payload = {
        "model_hash": model_hash,
        "features_hash": feat_hash_hex,
        "nonce": nonce,
        "context": context or {},
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _download_to_path(url: str, dst: Path, timeout_sec: int = 60):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    req = urllib.request.Request(url, headers={"User-Agent": "decision-integrity-proof-server/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as r:
        if getattr(r, "status", 200) != 200:
            raise RuntimeError(f"Download failed: HTTP {getattr(r, 'status', 'unknown')} for {url}")
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    tmp.replace(dst)


def _ensure_artifact(path: Path, url: str, label: str):
    if path.exists():
        return
    if not url:
        return
    try:
        _download_to_path(url, path)
        print(f"[startup] downloaded {label} -> {path} ({path.stat().st_size} bytes)")
    except Exception as e:
        print(f"[startup] failed downloading {label} from {url}: {e}")


def _maybe_fetch_missing_artifacts():
    # pk is required
    _ensure_artifact(EZKL_PK, EZKL_PK_URL, "pk.key")
    # srs optional download override (you already have it in repo, but allow url)
    _ensure_artifact(EZKL_SRS, EZKL_SRS_URL, "kzg17.srs")


def _check_ezkl_ready_or_raise():
    missing = []
    required = [EZKL_MODEL_ONNX, EZKL_SETTINGS, EZKL_COMPILED, EZKL_PK, EZKL_VK]
    for p in required:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EZKL artifacts missing",
                "missing": missing,
                "hint": (
                    "Put model.onnx/settings.json/model.ezkl/vk.key in ./ezkl_artifacts and "
                    "either provide pk.key on disk or set EZKL_PK_URL so the server downloads it."
                ),
            },
        )


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        return proc.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EZKL command failed",
                "cmd": cmd,
                "output": e.stdout,
            },
        )


def _write_ezkl_input_json(path: Path, features: List[float]):
    # Model expects shape [1,4]
    data = {"input_data": [features[:4]]}
    path.write_text(json.dumps(data))


def _extract_prediction_from_ezkl_witness(witness_json: Dict[str, Any]) -> float:
    for key in ("outputs", "output_data", "output"):
        if key in witness_json:
            out = witness_json[key]
            try:
                v = out[0][0] if isinstance(out, list) and isinstance(out[0], list) else out[0]
                return float(v)
            except Exception:
                pass
    return float("nan")


def _stage_ezkl_defaults_into_dir(dst_dir: Path) -> Path:
    """
    Render EZKL constraint: gen-witness expects default filenames in cwd.
    We stage artifacts into a temp dir with expected filenames:
      - model.compiled (from model.ezkl)
      - settings.json
      - model.onnx
      - pk.key / vk.key / kzg17.srs (also staged for convenience)
    """
    (dst_dir / "settings.json").write_bytes(EZKL_SETTINGS.read_bytes())

    compiled_local = dst_dir / "model.compiled"
    compiled_local.write_bytes(EZKL_COMPILED.read_bytes())

    (dst_dir / "model.onnx").write_bytes(EZKL_MODEL_ONNX.read_bytes())

    if EZKL_PK.exists():
        (dst_dir / "pk.key").write_bytes(EZKL_PK.read_bytes())
    if EZKL_VK.exists():
        (dst_dir / "vk.key").write_bytes(EZKL_VK.read_bytes())
    if EZKL_SRS.exists():
        (dst_dir / "kzg17.srs").write_bytes(EZKL_SRS.read_bytes())

    # Some builds look for /root/.ezkl/srs/kzg15.srs by default; try to populate if we can.
    try:
        srs_src = dst_dir / "kzg17.srs"
        if srs_src.exists():
            default_srs_dir = Path("/root/.ezkl/srs")
            default_srs_dir.mkdir(parents=True, exist_ok=True)
            (default_srs_dir / "kzg17.srs").write_bytes(srs_src.read_bytes())
            (default_srs_dir / "kzg15.srs").write_bytes(srs_src.read_bytes())
    except Exception:
        pass

    return compiled_local


# ---------- EZKL CLI bootstrap (handles Render versions missing gen-witness) ----------
def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def _run_no_throw(cmd: List[str]) -> str:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return (proc.stdout or "").strip()


def _ezkl_has_gen_witness(ezkl_bin: str) -> bool:
    out = _run_no_throw([ezkl_bin, "--help"]).lower()
    return "gen-witness" in out or "gen_witness" in out


def _install_ezkl_cli_into_tmp() -> str:
    install_url = os.getenv(
        "EZKL_INSTALL_SCRIPT_URL",
        "https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh",
    )

    tmp_dir = Path("/tmp/.ezkl_installer")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "install_ezkl_cli.sh"

    req = urllib.request.Request(install_url, headers={"User-Agent": "decision-integrity-proof-server/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        script_path.write_bytes(r.read())

    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    env = {**os.environ, "HOME": "/tmp"}
    proc = subprocess.run(
        ["bash", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"EZKL install script failed:\n{proc.stdout}")

    candidates = [
        Path("/tmp/.ezkl/ezkl"),
        Path("/tmp/.ezkl/ezkl.exe"),
        Path("/root/.ezkl/ezkl"),
        Path("/root/.ezkl/ezkl.exe"),
        Path("/tmp/.ezkl/ezkl-linux"),
    ]
    for c in candidates:
        if c.exists():
            c.chmod(c.stat().st_mode | stat.S_IEXEC)
            return str(c)

    found = _which("ezkl")
    if found:
        return found

    raise RuntimeError("EZKL install finished but ezkl binary not found in expected locations.")


def _ensure_working_ezkl_cli():
    global EZKL_BIN

    current = EZKL_BIN
    if not os.path.isabs(current):
        resolved = _which(current)
        if resolved:
            current = resolved

    if current and Path(current).exists() and _ezkl_has_gen_witness(current):
        EZKL_BIN = current
        print(f"[startup] ezkl ok: {EZKL_BIN} | version: {_run_no_throw([EZKL_BIN, '--version'])}")
        return

    print("[startup] ezkl missing gen-witness (or not found). Installing EZKL CLI into /tmp ...")
    installed = _install_ezkl_cli_into_tmp()

    if not _ezkl_has_gen_witness(installed):
        raise RuntimeError(f"Installed ezkl still missing gen-witness: {installed}")

    EZKL_BIN = installed
    print(f"[startup] using installed ezkl: {EZKL_BIN} | version: {_run_no_throw([EZKL_BIN, '--version'])}")


# ---------- Prove / Verify ----------
def ezkl_prove_real(model_hash: str, features: List[float]) -> Dict[str, Any]:
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)

        input_path = tdir / "input.json"
        witness_path = tdir / "witness.json"
        proof_path = tdir / "proof.json"

        _stage_ezkl_defaults_into_dir(tdir)

        _write_ezkl_input_json(input_path, features)

        # gen-witness (Render build expects defaults; try --data then fallback)
        try:
            _run([EZKL_BIN, "gen-witness", "--data", "input.json"], cwd=tdir)
        except HTTPException:
            _run([EZKL_BIN, "gen-witness"], cwd=tdir)

        if not witness_path.exists():
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "EZKL witness missing",
                    "hint": "gen-witness did not produce witness.json in the working directory",
                },
            )

        witness_obj = json.loads(witness_path.read_text())
        pred = _extract_prediction_from_ezkl_witness(witness_obj)

        # prove (use staged pk key if available)
        pk_local = tdir / "pk.key"
        pk_path = pk_local if pk_local.exists() else EZKL_PK.resolve()

        _run(
            [
                EZKL_BIN,
                "prove",
                "--witness",
                "witness.json",
                "--compiled-circuit",
                "model.compiled",
                "--pk-path",
                str(pk_path),
                "--proof-path",
                "proof.json",
            ],
            cwd=tdir,
        )

        if not proof_path.exists():
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "EZKL proof missing",
                    "hint": "prove did not produce proof.json in the working directory",
                },
            )

        proof_bytes = proof_path.read_bytes()

        feat_payload = json.dumps(features[:4], separators=(",", ":"), sort_keys=False).encode()
        fh = hashlib.sha256(feat_payload).digest()
        public_fingerprint = hashlib.sha256(model_hash.encode() + fh).digest()

        return {
            "proof_b64": base64.b64encode(proof_bytes).decode(),
            "public_inputs_b64": base64.b64encode(public_fingerprint).decode(),
            "prediction": pred,
        }


def ezkl_verify_real(proof_b64: str) -> bool:
    """
    IMPORTANT:
    Render's EZKL builds can be sensitive to filenames + cwd.
    We stage vk.key + model.compiled (+ settings/model/srs) into a temp cwd and
    run verify using RELATIVE paths in that cwd to avoid brittle path resolution.
    """
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        proof_path = tdir / "proof.json"
        proof_path.write_bytes(base64.b64decode(proof_b64.encode()))

        _stage_ezkl_defaults_into_dir(tdir)

        # If ezkl verify exits 0, it's valid. If not, _run throws.
        _run(
            [
                EZKL_BIN,
                "verify",
                "--proof-path",
                "proof.json",
                "--vk-path",
                "vk.key",
                "--compiled-circuit",
                "model.compiled",
            ],
            cwd=tdir,
        )
        return True


# =========================
# Startup checks
# =========================
@app.on_event("startup")
def _startup_checks():
    _ensure_working_ezkl_cli()
    _maybe_fetch_missing_artifacts()
    if STRICT_EZKL:
        _check_ezkl_ready_or_raise()


# =========================
# Basic endpoints
# =========================
@app.get("/health")
def health():
    return {"ok": True, "env": APP_ENV}


@app.get("/version")
def version():
    return {"version": "0.4.2-real-zk-ezkl"}


@app.get("/models")
def list_models():
    return [
        {
            "model_id": m["model_id"],
            "description": m.get("description", ""),
            "model_hash": m["model_hash"],
            "feature_schema": m.get("feature_schema", []),
            "policy_note": m.get("policy_note", ""),
        }
        for m in MODELS
    ]


# =========================
# Single-phase proof endpoints (REAL ZK)
# =========================
@app.post("/prove", response_model=ProveResponse)
def prove(req: ProveRequest):
    model = get_model(req.model_id)
    enforce_policy_constraints(req.features)

    zk = ezkl_prove_real(model["model_hash"], req.features)

    pred = zk.get("prediction")
    if pred is None or (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred))):
        pred = 0.0

    proof_b64 = zk["proof_b64"]
    if TAMPER_PROOF:
        proof_b64 = maybe_tamper_proof(proof_b64)

    return ProveResponse(
        model_id=req.model_id,
        model_hash=model["model_hash"],
        prediction=float(pred),
        proof_b64=proof_b64,
        public_inputs_b64=zk["public_inputs_b64"],
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    _ = get_model(req.model_id)
    try:
        valid = ezkl_verify_real(req.proof_b64)
        return VerifyResponse(model_id=req.model_id, valid=bool(valid))
    except HTTPException as e:
        # Log detailed EZKL failure info to Render logs, without changing API output
        try:
            print("[verify] EZKL failure detail:", json.dumps(e.detail, ensure_ascii=False))
        except Exception:
            print("[verify] EZKL failure detail (raw):", repr(e.detail))
        return VerifyResponse(model_id=req.model_id, valid=False)
    except Exception as e:
        print("[verify] unexpected exception:", repr(e))
        return VerifyResponse(model_id=req.model_id, valid=False)


# =========================
# Commitment endpoints
# =========================
@app.post("/commit", response_model=CommitResponse)
def commit(req: CommitRequest):
    model = get_model(req.model_id)
    payload = {
        "model_hash": model["model_hash"],
        "prediction": req.prediction,
        "nonce": req.nonce,
        "context": req.context or {},
    }
    commitment = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return CommitResponse(commitment_hash=f"sha256:{commitment}")


@app.post("/commit_case", response_model=CommitCaseResponse)
def commit_case(req: CommitCaseRequest):
    model = get_model(req.model_id)
    enforce_policy_constraints(req.features)

    fh = features_hash_hex(req.features[:4])
    commitment = compute_commitment(model["model_hash"], fh, req.nonce, req.context or {})

    # Option 2: store commit record (server-side, no API changes)
    _prune_commit_db()
    context_hash = hashlib.sha256(json.dumps(req.context or {}, sort_keys=True).encode()).hexdigest()
    with COMMIT_DB_LOCK:
        COMMIT_DB[commitment] = {
            "model_id": req.model_id,
            "model_hash_at_commit": model["model_hash"],
            "features_hash": fh,
            "nonce": req.nonce,
            "context_hash": context_hash,
            "created_at": time.time(),
        }

    return CommitCaseResponse(commitment_hash=commitment, features_hash=fh)


@app.post("/reveal", response_model=RevealResponse)
def reveal(req: RevealRequest):
    model = get_model(req.model_id)
    enforce_policy_constraints(req.features)

    fh = features_hash_hex(req.features[:4])

    # Compute commitment using CURRENT model hash (used only for fallback & returning commitment_hash)
    computed_now = compute_commitment(model["model_hash"], fh, req.nonce, req.context or {})

    # Option 2 enforcement:
    _prune_commit_db()
    with COMMIT_DB_LOCK:
        record = COMMIT_DB.get(req.expected_commitment_hash)

    if record:
        context_hash = hashlib.sha256(json.dumps(req.context or {}, sort_keys=True).encode()).hexdigest()

        model_hash_ok = (record.get("model_hash_at_commit") == model["model_hash"])
        features_ok = (record.get("features_hash") == fh)
        nonce_ok = (record.get("nonce") == req.nonce)
        context_ok = (record.get("context_hash") == context_hash)

        matches = bool(model_hash_ok and features_ok and nonce_ok and context_ok)
        commitment_to_return = req.expected_commitment_hash
    else:
        # Fallback behavior (e.g., server restarted and lost COMMIT_DB)
        matches = (computed_now == req.expected_commitment_hash)
        commitment_to_return = computed_now

    zk = ezkl_prove_real(model["model_hash"], req.features)

    pred = zk.get("prediction")
    if pred is None or (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred))):
        pred = 0.0

    proof_b64 = zk["proof_b64"]
    if TAMPER_PROOF:
        proof_b64 = maybe_tamper_proof(proof_b64)

    return RevealResponse(
        model_id=req.model_id,
        model_hash=model["model_hash"],
        prediction=float(pred),
        proof_b64=proof_b64,
        public_inputs_b64=zk["public_inputs_b64"],
        commitment_hash=commitment_to_return,
        commitment_matches=matches,
        features_hash=fh,
    )


# =========================
# Admin demo controls
# =========================
@app.post("/admin/tamper_proof")
def admin_tamper_proof(req: "AdminToggleRequest", x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    global TAMPER_PROOF
    TAMPER_PROOF = req.enabled
    return {"tamper_proof": TAMPER_PROOF}


@app.post("/admin/set_model_hash")
def admin_set_model_hash(req: "AdminSetModelHashRequest", x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    for m in MODELS:
        if m["model_id"] == req.model_id:
            m["model_hash"] = req.model_hash
            return {"ok": True, "model_id": req.model_id, "model_hash": req.model_hash}
    raise HTTPException(status_code=404, detail="Unknown model_id")


@app.post("/admin/reset_demo")
def admin_reset_demo(x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    global TAMPER_PROOF
    TAMPER_PROOF = False
    for m in MODELS:
        if m["model_id"] == "geo_escalation_7d_v1":
            m["model_hash"] = DEFAULT_MODEL_HASH
    return {
        "ok": True,
        "tamper_proof": TAMPER_PROOF,
        "model_id": "geo_escalation_7d_v1",
        "model_hash": get_model("geo_escalation_7d_v1")["model_hash"],
    }
