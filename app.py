# app.py
"""
Decision Integrity Proof Server (Real ZK via EZKL CLI)

Key points:
- We install the EZKL CLI (ezkl) and shell out to it for gen-witness / prove / verify.
- Artifacts are generated OFFLINE and served from disk (or downloaded from GitHub Releases).
- This is a demo (not production hardening).
"""

import base64
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_ENV = os.getenv("APP_ENV", "dev")
ADMIN_KEY = os.getenv("ADMIN_KEY", "dev-admin-key")

# Demo toggle: intentionally corrupt proof transport to show verification failures
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
EZKL_MODEL_ONNX = Path(os.getenv("EZKL_MODEL_ONNX", str(EZKL_ARTIFACTS_DIR / "model.onnx")))
EZKL_SETTINGS = Path(os.getenv("EZKL_SETTINGS", str(EZKL_ARTIFACTS_DIR / "settings.json")))
EZKL_COMPILED = Path(os.getenv("EZKL_COMPILED", str(EZKL_ARTIFACTS_DIR / "model.ezkl")))
EZKL_PK = Path(os.getenv("EZKL_PK", str(EZKL_ARTIFACTS_DIR / "pk.key")))
EZKL_VK = Path(os.getenv("EZKL_VK", str(EZKL_ARTIFACTS_DIR / "vk.key")))
# You generated kzg17.srs in Colab; keep that name.
EZKL_SRS = Path(os.getenv("EZKL_SRS", str(EZKL_ARTIFACTS_DIR / "kzg17.srs")))

# Optional: download artifacts at runtime if missing
EZKL_MODEL_ONNX_URL = os.getenv("EZKL_MODEL_ONNX_URL", "")
EZKL_SETTINGS_URL = os.getenv("EZKL_SETTINGS_URL", "")
EZKL_COMPILED_URL = os.getenv("EZKL_COMPILED_URL", "")
EZKL_PK_URL = os.getenv("EZKL_PK_URL", "")
EZKL_VK_URL = os.getenv("EZKL_VK_URL", "")
EZKL_SRS_URL = os.getenv("EZKL_SRS_URL", "")

STRICT_EZKL = os.getenv("STRICT_EZKL", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Decision Integrity Proof Server", version="0.4.1-real-zk-ezkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Request/Response Models
# =========================

class ProveRequest(BaseModel):
    model_id: str = Field(..., description="Which approved model version to use")
    features: List[float] = Field(..., description="Numeric feature vector (first 4 used)")
    context: Optional[Dict[str, Any]] = Field(default=None)


class ProveResponse(BaseModel):
    model_id: str
    model_hash: str
    prediction: float
    proof_b64: str
    public_inputs_b64: str


class VerifyRequest(BaseModel):
    model_id: str
    proof_b64: str
    public_inputs_b64: str


class VerifyResponse(BaseModel):
    model_id: str
    valid: bool


class CommitRequest(BaseModel):
    model_id: str
    prediction: float
    nonce: str
    context: Optional[Dict[str, Any]] = None


class CommitResponse(BaseModel):
    commitment_hash: str


class CommitCaseRequest(BaseModel):
    model_id: str
    features: List[float]
    nonce: str
    context: Optional[Dict[str, Any]] = None


class CommitCaseResponse(BaseModel):
    commitment_hash: str
    features_hash: str


class RevealRequest(BaseModel):
    model_id: str
    features: List[float]
    nonce: str
    context: Optional[Dict[str, Any]] = None
    expected_commitment_hash: str


class RevealResponse(BaseModel):
    model_id: str
    model_hash: str
    prediction: float
    proof_b64: str
    public_inputs_b64: str
    commitment_hash: str
    commitment_matches: bool
    features_hash: str


class AdminToggleRequest(BaseModel):
    enabled: bool


class AdminSetModelHashRequest(BaseModel):
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
            detail={"error": "EZKL command failed", "cmd": cmd, "output": e.stdout},
        )


def _curl_download(url: str, dst: Path):
    if not url:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Use curl via subprocess to avoid adding python requests dependency.
    _run(["curl", "-fL", url, "-o", str(dst)])


def _ensure_artifacts_present():
    """
    If artifacts are missing but *_URL env vars are provided, download them.
    This makes Render deploys easier (you can store big pk.key in GitHub Releases).
    """
    EZKL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    mapping = [
        (EZKL_MODEL_ONNX, EZKL_MODEL_ONNX_URL),
        (EZKL_SETTINGS, EZKL_SETTINGS_URL),
        (EZKL_COMPILED, EZKL_COMPILED_URL),
        (EZKL_PK, EZKL_PK_URL),
        (EZKL_VK, EZKL_VK_URL),
        (EZKL_SRS, EZKL_SRS_URL),
    ]

    for path, url in mapping:
        if not path.exists() and url:
            _curl_download(url, path)


def _ensure_ezkl_binary():
    """
    Ensure EZKL_BIN is executable and exists in PATH or as an absolute path.
    """
    # If EZKL_BIN is a path, check it
    if "/" in EZKL_BIN:
        p = Path(EZKL_BIN)
        if not p.exists():
            raise HTTPException(status_code=500, detail={"error": "EZKL binary not found", "path": str(p)})
        st = p.stat()
        if not (st.st_mode & stat.S_IXUSR):
            p.chmod(st.st_mode | stat.S_IXUSR)
        return

    # Otherwise rely on PATH
    try:
        _run([EZKL_BIN, "--version"])
    except HTTPException:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EZKL CLI not found on PATH",
                "hint": "Set EZKL_BIN to an absolute path like /usr/local/bin/ezkl",
            },
        )


def _ensure_srs_in_default_location():
    """
    In your Colab run (ezkl 23.0.3), EZKL expects kzg17.srs (and sometimes kzg15.srs)
    under ~/.ezkl/ . We copy your pinned SRS there if present.
    """
    if not EZKL_SRS.exists():
        return

    home = Path(os.getenv("HOME", "/root"))
    default_dir = home / ".ezkl"
    default_dir.mkdir(parents=True, exist_ok=True)

    # Copy to the expected filenames
    dst17 = default_dir / "kzg17.srs"
    if not dst17.exists():
        shutil.copy2(EZKL_SRS, dst17)

    # Some flows look for kzg15.srs too; copy same file (good enough for this demo)
    dst15 = default_dir / "kzg15.srs"
    if not dst15.exists():
        shutil.copy2(EZKL_SRS, dst15)


def _check_ezkl_ready_or_raise():
    missing = []
    for p in [EZKL_MODEL_ONNX, EZKL_SETTINGS, EZKL_COMPILED, EZKL_PK, EZKL_VK]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EZKL artifacts missing",
                "missing": missing,
                "hint": (
                    "Provide artifacts on disk OR set EZKL_*_URL env vars so the server downloads them. "
                    "Required: model.onnx, settings.json, model.ezkl, pk.key, vk.key. "
                    "Recommended: kzg17.srs (set EZKL_SRS_URL)."
                ),
            },
        )


def _write_ezkl_input_json(path: Path, features: List[float]):
    # Your ONNX expects [1,4]
    data = {"input_data": [features[:4]]}
    path.write_text(json.dumps(data))


def _extract_prediction_from_ezkl_witness(witness_json: Dict[str, Any]) -> float:
    """
    Best-effort extraction; different ezkl versions may structure witness differently.
    """
    for key in ("outputs", "output_data", "output"):
        if key in witness_json:
            out = witness_json[key]
            try:
                if isinstance(out, list) and len(out) > 0:
                    if isinstance(out[0], list) and len(out[0]) > 0:
                        return float(out[0][0])
                    return float(out[0])
            except Exception:
                pass
    return float("nan")


def ezkl_prove_real(model_hash: str, features: List[float]) -> Dict[str, Any]:
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        input_path = tdir / "input.json"
        witness_path = tdir / "witness.json"
        proof_path = tdir / "proof.json"

        _write_ezkl_input_json(input_path, features)

        # 1) gen-witness (runs inference)
        _run(
            [
                EZKL_BIN,
                "gen-witness",
                "--data",
                str(input_path),
                "--model",
                str(EZKL_MODEL_ONNX),
                "--output",
                str(witness_path),
            ],
            cwd=tdir,
        )

        witness_obj = json.loads(witness_path.read_text())
        pred = _extract_prediction_from_ezkl_witness(witness_obj)

        # 2) prove
        _run(
            [
                EZKL_BIN,
                "prove",
                "--witness",
                str(witness_path),
                "--compiled-circuit",
                str(EZKL_COMPILED),
                "--pk-path",
                str(EZKL_PK),
                "--proof-path",
                str(proof_path),
            ],
            cwd=tdir,
        )

        proof_bytes = proof_path.read_bytes()

        # UI/audit fingerprint (not required by EZKL verify)
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

        out = _run(
            [
                EZKL_BIN,
                "verify",
                "--proof-path",
                str(proof_path),
                "--vk-path",
                str(EZKL_VK),
                "--compiled-circuit",
                str(EZKL_COMPILED),
            ],
            cwd=tdir,
        )

        low = out.lower()
        return ("verified" in low and "true" in low) or ("valid" in low and "true" in low) or ("success" in low)

# =========================
# Startup checks
# =========================

@app.on_event("startup")
def _startup_checks():
    _ensure_ezkl_binary()
    _ensure_artifacts_present()
    _ensure_srs_in_default_location()

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
    # also show ezkl version for debugging
    try:
        v = _run([EZKL_BIN, "--version"]).strip()
    except Exception:
        v = "unknown"
    return {"version": "0.4.1-real-zk-ezkl", "ezkl": v}

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
# Proof endpoints
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
def admin_tamper_proof(req: AdminToggleRequest, x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    global TAMPER_PROOF
    TAMPER_PROOF = req.enabled
    return {"tamper_proof": TAMPER_PROOF}

@app.post("/admin/set_model_hash")
def admin_set_model_hash(req: AdminSetModelHashRequest, x_admin_key: Optional[str] = Header(None)):
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
