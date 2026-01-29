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

EZKL_MODEL_ONNX = Path(os.getenv("EZKL_MODEL_ONNX", str(EZKL_ARTIFACTS_DIR / "model.onnx")))
EZKL_SETTINGS = Path(os.getenv("EZKL_SETTINGS", str(EZKL_ARTIFACTS_DIR / "settings.json")))
EZKL_COMPILED = Path(os.getenv("EZKL_COMPILED", str(EZKL_ARTIFACTS_DIR / "model.ezkl")))
EZKL_PK = Path(os.getenv("EZKL_PK", str(EZKL_ARTIFACTS_DIR / "pk.key")))
EZKL_VK = Path(os.getenv("EZKL_VK", str(EZKL_ARTIFACTS_DIR / "vk.key")))

# Optional SRS (not always needed at verify time, but good to pin for reproducibility)
EZKL_SRS = Path(os.getenv("EZKL_SRS", str(EZKL_ARTIFACTS_DIR / "kzg17.srs")))

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


def ezkl_prove_real(model_hash: str, features: List[float]) -> Dict[str, Any]:
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        input_path = tdir / "input.json"
        witness_path = tdir / "witness.json"
        proof_path = tdir / "proof.json"

        _write_ezkl_input_json(input_path, features)

        # 1) Witness
        # Your Render EZKL version does NOT support:
        #   ezkl gen-witness --data <DATA> --model <MODEL> --output <OUT>
        # It expects:
        #   ezkl gen-witness --data <DATA>
        # and relies on default artifact filenames in the current working directory.
        #
        # So we copy the artifacts into this temp dir and run gen-witness there.
        (tdir / "model.onnx").write_bytes(EZKL_MODEL_ONNX.read_bytes())
        (tdir / "settings.json").write_bytes(EZKL_SETTINGS.read_bytes())
        (tdir / "model.ezkl").write_bytes(EZKL_COMPILED.read_bytes())

        _run(
            [
                EZKL_BIN,
                "gen-witness",
                "--data",
                str(input_path),
            ],
            cwd=tdir,
        )

        witness_obj = json.loads(witness_path.read_text())
        pred = _extract_prediction_from_ezkl_witness(witness_obj)

        # 2) Prove
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

        # Optional "public fingerprint" for your UI
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
    # Try downloading pk.key (and optional SRS) BEFORE strict checks
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
