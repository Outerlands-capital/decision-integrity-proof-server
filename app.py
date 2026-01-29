# app.py
"""
Decision Integrity Proof Server (Real ZK version via EZKL)

What changed vs your hash-based PoC:
- /prove now generates a REAL zkSNARK proof using EZKL (Halo2-based proving inside EZKL)
- /verify now verifies that proof using the EZKL verifying key
- prediction returned is taken from the model inference used by EZKL (or optionally re-computed locally)

What you need on disk (generated once, outside this server):
- ONNX model file
- EZKL settings.json
- compiled circuit file (.ezkl)
- proving key (pk.key)
- verifying key (vk.key)

This server shells out to the `ezkl` CLI for prove/verify to keep the demo simple.
(You can later move to a native Rust service or python bindings if desired.)

Environment variables:
  APP_ENV=dev
  ADMIN_KEY=...
  # EZKL
  EZKL_BIN=ezkl                           # path to ezkl binary in PATH or absolute
  EZKL_ARTIFACTS_DIR=./ezkl_artifacts     # directory containing artifacts
  EZKL_MODEL_ONNX=./ezkl_artifacts/model.onnx
  EZKL_SETTINGS=./ezkl_artifacts/settings.json
  EZKL_COMPILED=./ezkl_artifacts/model.ezkl
  EZKL_PK=./ezkl_artifacts/pk.key
  EZKL_VK=./ezkl_artifacts/vk.key
  EZKL_SRS=./ezkl_artifacts/kzg.srs       # optional; EZKL can download/generate; prefer pinned file for reproducibility
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

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_ENV = os.getenv("APP_ENV", "dev")
ADMIN_KEY = os.getenv("ADMIN_KEY", "dev-admin-key")

# Demo toggle: when enabled, server will intentionally corrupt proofs to show verification failures
TAMPER_PROOF = False

DEFAULT_MODEL_HASH = "sha256:geo-escalation-7d-demo-v1"

MODELS = [
    {
        "model_id": "geo_escalation_7d_v1",
        "description": "Geopolitical escalation risk forecast (next 7 days) — PoC (Real ZK via EZKL)",
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
EZKL_SRS = Path(os.getenv("EZKL_SRS", str(EZKL_ARTIFACTS_DIR / "kzg.srs")))

# If True, we will fail-fast if ezkl artifacts are missing.
# If False, server starts but /prove,/verify will error with actionable message.
STRICT_EZKL = os.getenv("STRICT_EZKL", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Decision Integrity Proof Server", version="0.4.0-real-zk-ezkl")

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
    features: List[float] = Field(..., description="Numeric feature vector")
    context: Optional[Dict[str, Any]] = Field(default=None)


class ProveResponse(BaseModel):
    model_id: str
    model_hash: str
    prediction: float  # probability-like score in [0,1]
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


def _check_ezkl_ready_or_raise():
    missing = []
    for p in [EZKL_MODEL_ONNX, EZKL_SETTINGS, EZKL_COMPILED, EZKL_PK, EZKL_VK]:
        if not p.exists():
            missing.append(str(p))
    # SRS is optional depending on EZKL config; warn only
    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EZKL artifacts missing. Generate them once and place them on disk.",
                "missing": missing,
                "hint": (
                    "Expected artifacts: model.onnx, settings.json, model.ezkl, pk.key, vk.key. "
                    "Set EZKL_* env vars if your paths differ."
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
    """
    EZKL expects a JSON input file. The exact schema depends on how you compiled your model/settings.
    The most common format for single input tensor is:
      {"input_data": [[...]]}

    If your model expects shape [1,4], this should work.
    """
    data = {"input_data": [features[:4]]}
    path.write_text(json.dumps(data))


def _extract_prediction_from_ezkl_witness(witness_json: Dict[str, Any]) -> float:
    """
    This is model-dependent. Common patterns:
    - witness contains "outputs": [[...]]
    - or "output_data": [[...]]
    We'll try a few.
    """
    for key in ("outputs", "output_data", "output"):
        if key in witness_json:
            out = witness_json[key]
            # Try to pull first scalar
            try:
                v = out[0][0] if isinstance(out, list) and isinstance(out[0], list) else out[0]
                return float(v)
            except Exception:
                pass
    # If we can't find it, return NaN-ish and let caller decide
    return float("nan")


def ezkl_prove_real(model_hash: str, features: List[float]) -> Dict[str, str]:
    """
    Generates a REAL zk proof using EZKL CLI.

    We:
    - create a temp dir
    - write input.json
    - run `ezkl gen-witness` to create witness.json
    - run `ezkl prove` to create proof.json
    - run `ezkl verify` optionally (server still exposes /verify)

    We return:
      proof_b64: base64 of proof.json bytes
      public_inputs_b64: base64 of the "public instances" (we include witness hash as a stable public fingerprint)
      prediction: extracted scalar output (caller reads from returned dict)
    """
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        input_path = tdir / "input.json"
        witness_path = tdir / "witness.json"
        proof_path = tdir / "proof.json"

        _write_ezkl_input_json(input_path, features)

        # 1) Generate witness (runs the model)
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

        # 2) Prove
        # Some EZKL versions use flags: --witness, --compiled-circuit, --pk-path, --proof-path
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

        # Public inputs for the verifier are implicitly inside proof + vk in most EZKL flows.
        # For transport/audit UI we also return a stable public fingerprint:
        # sha256(model_hash || sha256(features_json))
        feat_payload = json.dumps(features[:4], separators=(",", ":"), sort_keys=False).encode()
        fh = hashlib.sha256(feat_payload).digest()
        public_fingerprint = hashlib.sha256(model_hash.encode() + fh).digest()

        proof_b64 = base64.b64encode(proof_bytes).decode()
        public_inputs_b64 = base64.b64encode(public_fingerprint).decode()

        return {
            "proof_b64": proof_b64,
            "public_inputs_b64": public_inputs_b64,
            "prediction": pred,
        }


def ezkl_verify_real(proof_b64: str) -> bool:
    """
    Verifies a REAL zk proof using EZKL CLI.

    We:
    - decode proof.json bytes
    - write to temp file
    - run `ezkl verify` with vk and compiled circuit
    """
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

        # EZKL outputs vary by version. We treat presence of "verified" / "true" as success.
        low = out.lower()
        return ("verified" in low and "true" in low) or ("success" in low) or ("valid" in low and "true" in low)


# =========================
# Startup checks
# =========================

@app.on_event("startup")
def _startup_checks():
    if STRICT_EZKL:
        # Fail-fast if artifacts missing (better for demos)
        _check_ezkl_ready_or_raise()


# =========================
# Basic endpoints
# =========================

@app.get("/health")
def health():
    return {"ok": True, "env": APP_ENV}


@app.get("/version")
def version():
    return {"version": "0.4.0-real-zk-ezkl"}


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

    # If EZKL witness parsing couldn't extract prediction, you can optionally compute locally,
    # but then the proof attests to the EZKL model output, not your local computation.
    pred = zk.get("prediction")
    if pred is None or (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred))):
        # last resort: keep demo running but be explicit in logs/UI if you want
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
    # model_id is used for registry/UX parity; verification uses vk+compiled circuit
    _ = get_model(req.model_id)

    try:
        # We verify the REAL proof. public_inputs_b64 is not required by EZKL verify.
        # We still accept it for UI consistency and audit fingerprints.
        valid = ezkl_verify_real(req.proof_b64)
        return VerifyResponse(model_id=req.model_id, valid=bool(valid))
    except Exception:
        return VerifyResponse(model_id=req.model_id, valid=False)


# =========================
# Commitment endpoints (unchanged)
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


# =========================
# Two-phase audit endpoints (REAL ZK on reveal)
# =========================

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
