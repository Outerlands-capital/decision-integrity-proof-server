# app.py
"""
Decision Integrity Proof Server (Real ZK version via EZKL)

- /prove generates a REAL zk proof using EZKL
- /verify verifies it using vk.key

Artifacts:
- model.onnx, settings.json, model.ezkl, vk.key are in repo under ./ezkl_artifacts
- pk.key is LARGE, so it is downloaded at runtime from EZKL_PK_URL if missing

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
"""

import base64
import hashlib
import json
import math
import os
import subprocess
import tempfile
import stat
import platform

from pathlib import Path
from typing import Any, Dict, List, Optional

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

# -------------------------
# EZKL configuration
# -------------------------
EZKL_BIN = os.getenv("EZKL_BIN", "ezkl")
EZKL_ARTIFACTS_DIR = Path(os.getenv("EZKL_ARTIFACTS_DIR", "./ezkl_artifacts"))

EZKL_ARTIFACTS_DIR = Path(os.getenv("EZKL_ARTIFACTS_DIR", "./ezkl_artifacts")).resolve()

EZKL_MODEL_ONNX = Path(os.getenv("EZKL_MODEL_ONNX", str(EZKL_ARTIFACTS_DIR / "model.onnx"))).resolve()
EZKL_SETTINGS   = Path(os.getenv("EZKL_SETTINGS",   str(EZKL_ARTIFACTS_DIR / "settings.json"))).resolve()
EZKL_COMPILED   = Path(os.getenv("EZKL_COMPILED",   str(EZKL_ARTIFACTS_DIR / "model.ezkl"))).resolve()
EZKL_PK         = Path(os.getenv("EZKL_PK",         str(EZKL_ARTIFACTS_DIR / "pk.key"))).resolve()
EZKL_VK         = Path(os.getenv("EZKL_VK",         str(EZKL_ARTIFACTS_DIR / "vk.key"))).resolve()
EZKL_SRS        = Path(os.getenv("EZKL_SRS",        str(EZKL_ARTIFACTS_DIR / "kzg17.srs"))).resolve()


# URLs for missing artifacts (pk is the important one for you)
EZKL_PK_URL = os.getenv("EZKL_PK_URL", "").strip()
EZKL_SRS_URL = os.getenv("EZKL_SRS_URL", "").strip()

# Fail-fast on startup if required artifacts are missing (after attempting downloads)
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
# Request/Response Models
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
        if r.status != 200:
            raise RuntimeError(f"Download failed: HTTP {r.status} for {url}")
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
    # Only pk is required in your setup (others are in repo)
    _ensure_artifact(EZKL_PK, EZKL_PK_URL, "pk.key")
    # SRS optional, but if you want to pin it, set EZKL_SRS_URL
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

    return compiled_local



def ezkl_prove_real(model_hash: str, features: List[float]) -> Dict[str, Any]:
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)

        input_path = tdir / "input.json"
        witness_path = tdir / "witness.json"
        proof_path = tdir / "proof.json"

        # Stage artifacts in the temp dir under default filenames expected by EZKL CLI on Render
        compiled_local = _stage_ezkl_defaults_into_dir(tdir)

        # Write inputs (EZKL reads input.json from cwd in some builds; we keep it in cwd)
        _write_ezkl_input_json(input_path, features)

        # 1) Witness (Render EZKL expects defaults in cwd; flags vary by build)
        # Try with --data first, then fallback to bare command if unsupported.
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

        # 2) Prove
        # IMPORTANT: use an absolute pk path (or staged pk.key in cwd) so cwd doesn't break it.
        pk_local = tdir / "pk.key"
        pk_path = pk_local if pk_local.exists() else EZKL_PK.resolve()

        # Some builds may also want srs in cwd; if staged, it'll be there.
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

        # Optional "public fingerprint" for your UI (kept exactly as your current logic)
        feat_payload = json.dumps(features[:4], separators=(",", ":"), sort_keys=False).encode()
        fh = hashlib.sha256(feat_payload).digest()
        public_fingerprint = hashlib.sha256(model_hash.encode() + fh).digest()

        return {
            "proof_b64": base64.b64encode(proof_bytes).decode(),
            "public_inputs_b64": base64.b64encode(public_fingerprint).decode(),
            "prediction": pred,
        }



def ezkl_verify_real(proof_b64: str) -> bool:
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        proof_path = tdir / "proof.json"
        proof_path.write_bytes(base64.b64decode(proof_b64.encode()))

        # Stage the compiled circuit under the default filename; also use it explicitly in flags.
        compiled_local = _stage_ezkl_defaults_into_dir(tdir)

        out = _run(
            [
                EZKL_BIN,
                "verify",
                "--proof-path",
                str(proof_path),
                "--vk-path",
                str(EZKL_VK),
                "--compiled-circuit",
                str(compiled_local),
            ],
            cwd=tdir,
        )

        low = out.lower()
        return ("verified" in low and "true" in low) or ("valid" in low and "true" in low) or ("success" in low)

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
    # help text varies by version; this is a decent heuristic
    return "gen-witness" in out or "gen_witness" in out

def _install_ezkl_cli_into_tmp() -> str:
    """
    Installs EZKL CLI to /tmp/.ezkl using the official install script,
    and returns the absolute path to the installed ezkl binary.

    This keeps the fix strictly server-side (app.py only).
    """
    install_url = os.getenv(
        "EZKL_INSTALL_SCRIPT_URL",
        "https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh",
    )

    tmp_dir = Path("/tmp/.ezkl_installer")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "install_ezkl_cli.sh"

    # Download script
    req = urllib.request.Request(install_url, headers={"User-Agent": "decision-integrity-proof-server/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        script_path.write_bytes(r.read())

    # Make executable
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    # Run installer (it typically installs into /root/.ezkl or ~/.ezkl depending on env)
    # We force HOME to /tmp to avoid permission/path weirdness on Render.
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

    # Common install locations
    candidates = [
        Path("/tmp/.ezkl/ezkl"),       # if script uses $HOME/.ezkl
        Path("/tmp/.ezkl/ezkl.exe"),
        Path("/root/.ezkl/ezkl"),      # sometimes script uses /root/.ezkl even if HOME overridden
        Path("/root/.ezkl/ezkl.exe"),
        Path("/tmp/.ezkl/ezkl-linux"), # just in case
    ]
    for c in candidates:
        if c.exists():
            c.chmod(c.stat().st_mode | stat.S_IEXEC)
            return str(c)

    # Fallback: look for ezkl on PATH after install
    found = _which("ezkl")
    if found:
        return found

    raise RuntimeError("EZKL install finished but ezkl binary not found in expected locations.")

def _ensure_working_ezkl_cli():
    """
    Ensures EZKL_BIN points to a CLI that supports `gen-witness`.
    If the system ezkl is missing it, install a proper one into /tmp and use that.
    """
    global EZKL_BIN

    # If EZKL_BIN is a bare name, try resolve it
    current = EZKL_BIN
    if not os.path.isabs(current):
        resolved = _which(current)
        if resolved:
            current = resolved

    # If current binary works, keep it
    if current and Path(current).exists() and _ezkl_has_gen_witness(current):
        EZKL_BIN = current
        print(f"[startup] ezkl ok: {EZKL_BIN} | version: {_run_no_throw([EZKL_BIN, '--version'])}")
        return

    # Otherwise install
    print(f"[startup] ezkl missing gen-witness (or not found). Installing EZKL CLI into /tmp ...")
    installed = _install_ezkl_cli_into_tmp()

    if not _ezkl_has_gen_witness(installed):
        raise RuntimeError(f"Installed ezkl still missing gen-witness: {installed}")

    EZKL_BIN = installed
    print(f"[startup] using installed ezkl: {EZKL_BIN} | version: {_run_no_throw([EZKL_BIN, '--version'])}")


# =========================
# Startup checks
# =========================

@app.on_event("startup")
def _startup_checks():
    # 0) Ensure we have an EZKL CLI that actually supports gen-witness
    _ensure_working_ezkl_cli()

    # 1) Try downloading pk.key (and optional SRS) BEFORE strict checks
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
    except Exception:
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
    return CommitCaseResponse(commitment_hash=commitment, features_hash=fh)


@app.post("/reveal", response_model=RevealResponse)
def reveal(req: RevealRequest):
    model = get_model(req.model_id)
    enforce_policy_constraints(req.features)

    fh = features_hash_hex(req.features[:4])
    commitment = compute_commitment(model["model_hash"], fh, req.nonce, req.context or {})
    matches = (commitment == req.expected_commitment_hash)

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
        commitment_hash=commitment,
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
