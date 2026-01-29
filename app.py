# app.py
"""
Decision Integrity Proof Server (Real ZK via EZKL 23.0.3)

Render runs only:
  gen-witness -> prove -> verify

Offline (Colab) produced:
  model.ezkl, settings.json, model.onnx, vk.key, kzg17.srs, pk.key

We invoke EZKL via CLI: ezkl <cmd>
Dockerfile installs ezkl into /usr/local/bin/ezkl.
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

# Demo toggle: corrupt proofs intentionally to show verify failure.
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

EZKL_ARTIFACTS_DIR = Path(os.getenv("EZKL_ARTIFACTS_DIR", "/app/ezkl_artifacts"))

# Runtime-required
EZKL_COMPILED = Path(os.getenv("EZKL_COMPILED", str(EZKL_ARTIFACTS_DIR / "model.ezkl")))
EZKL_PK = Path(os.getenv("EZKL_PK", str(EZKL_ARTIFACTS_DIR / "pk.key")))
EZKL_VK = Path(os.getenv("EZKL_VK", str(EZKL_ARTIFACTS_DIR / "vk.key")))

# Optional (but typically present in your repo)
EZKL_MODEL_ONNX = Path(os.getenv("EZKL_MODEL_ONNX", str(EZKL_ARTIFACTS_DIR / "model.onnx")))
EZKL_SETTINGS = Path(os.getenv("EZKL_SETTINGS", str(EZKL_ARTIFACTS_DIR / "settings.json")))
EZKL_SRS = Path(os.getenv("EZKL_SRS", str(EZKL_ARTIFACTS_DIR / "kzg17.srs")))

STRICT_EZKL = os.getenv("STRICT_EZKL", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Decision Integrity Proof Server", version="0.4.3-real-zk-ezkl-23.0.3")

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
    detail: Optional[Any] = None


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
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Command not found",
                "cmd": cmd,
                "hint": "Check that ezkl is installed in the container and EZKL_BIN is correct.",
            },
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Command failed",
                "cmd": cmd,
                "output": e.stdout,
            },
        )


def _run_ezkl(args: List[str], cwd: Optional[Path] = None) -> str:
    return _run([EZKL_BIN] + args, cwd=cwd)


def _artifact_info(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {"exists": False, "path": str(p)}
    try:
        size = p.stat().st_size
    except Exception:
        size = None
    return {"exists": True, "size_bytes": size, "path": str(p)}


def _check_ezkl_ready_or_raise():
    missing = []
    for p in [EZKL_COMPILED, EZKL_PK, EZKL_VK]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EZKL artifacts missing (runtime-required).",
                "missing": missing,
                "hint": "Runtime requires: model.ezkl, pk.key, vk.key in /app/ezkl_artifacts.",
            },
        )


def _write_ezkl_input_json(path: Path, features: List[float]):
    # Matches your Colab format:
    # {"input_data": [[f1,f2,f3,f4]]}
    data = {"input_data": [features[:4]]}
    path.write_text(json.dumps(data, separators=(",", ":")))


def _find_first_scalar(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, list) and x:
        for item in x:
            v = _find_first_scalar(item)
            if v is not None:
                return v
    if isinstance(x, dict):
        for k in ("outputs", "output_data", "output", "out", "result"):
            if k in x:
                v = _find_first_scalar(x[k])
                if v is not None:
                    return v
        for v0 in x.values():
            v = _find_first_scalar(v0)
            if v is not None:
                return v
    return None


def _extract_prediction_from_witness_file(witness_path: Path) -> float:
    try:
        obj = json.loads(witness_path.read_text())
    except Exception:
        return float("nan")
    v = _find_first_scalar(obj)
    return float(v) if v is not None else float("nan")


def ezkl_prove_real(model_hash: str, features: List[float]) -> Dict[str, Any]:
    """
    EZKL 23.0.3 flow (compiled circuit):
      gen-witness --data input.json --compiled-circuit model.ezkl --output witness.json
      prove       --witness witness.json --compiled-circuit model.ezkl --pk-path pk.key --proof-path proof.json
    """
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        input_path = tdir / "input.json"
        witness_path = tdir / "witness.json"
        proof_path = tdir / "proof.json"

        _write_ezkl_input_json(input_path, features)

        _run_ezkl(
            [
                "gen-witness",
                "--data",
                str(input_path),
                "--compiled-circuit",
                str(EZKL_COMPILED),
                "--output",
                str(witness_path),
            ],
            cwd=tdir,
        )

        pred = _extract_prediction_from_witness_file(witness_path)

        _run_ezkl(
            [
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

        # Stable UI fingerprint (not required by EZKL verify)
        feat_payload = json.dumps(features[:4], separators=(",", ":"), sort_keys=False).encode()
        fh = hashlib.sha256(feat_payload).digest()
        public_fingerprint = hashlib.sha256(model_hash.encode() + fh).digest()

        return {
            "proof_b64": base64.b64encode(proof_bytes).decode(),
            "public_inputs_b64": base64.b64encode(public_fingerprint).decode(),
            "prediction": pred,
        }


def ezkl_verify_real(proof_b64: str) -> Dict[str, Any]:
    """
    EZKL 23.0.3 verify:
      verify --proof-path proof.json --vk-path vk.key --compiled-circuit model.ezkl
    """
    _check_ezkl_ready_or_raise()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        proof_path = tdir / "proof.json"

        try:
            proof_path.write_bytes(base64.b64decode(proof_b64.encode()))
        except Exception as e:
            return {"valid": False, "detail": {"error": "Invalid base64 proof", "exception": str(e)}}

        out = _run_ezkl(
            [
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
        valid = ("verified" in low and "true" in low) or ("valid" in low and "true" in low) or ("success" in low)
        return {"valid": bool(valid), "detail": {"output": out}}


# =========================
# Startup checks
# =========================

@app.on_event("startup")
def _startup_checks():
    if STRICT_EZKL:
        _check_ezkl_ready_or_raise()
        _ = _run_ezkl(["--version"])


# =========================
# Basic endpoints
# =========================

@app.get("/health")
def health():
    ezkl_version = None
    try:
        ezkl_version = _run_ezkl(["--version"]).strip()
    except Exception:
        ezkl_version = None

    return {
        "ok": True,
        "env": APP_ENV,
        "ezkl_version": ezkl_version,
        "artifacts": {
            "compiled": _artifact_info(EZKL_COMPILED),
            "pk": _artifact_info(EZKL_PK),
            "vk": _artifact_info(EZKL_VK),
            "srs": _artifact_info(EZKL_SRS),
            "onnx": _artifact_info(EZKL_MODEL_ONNX),
            "settings": _artifact_info(EZKL_SETTINGS),
        },
    }


@app.get("/version")
def version():
    return {"version": "0.4.3-real-zk-ezkl-23.0.3"}


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
# Single-phase proof endpoints
# =========================

@app.post("/prove", response_model=ProveResponse)
def prove(req: ProveRequest):
    model = get_model(req.model_id)
    enforce_policy_constraints(req.features)

    zk = ezkl_prove_real(model["model_hash"], req.features)

    pred = zk.get("prediction", float("nan"))
    if isinstance(pred, (int, float)) and (math.isnan(pred) or math.isinf(pred)):
        pred = 0.0  # deterministic fallback for demo

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
    res = ezkl_verify_real(req.proof_b64)
    return VerifyResponse(model_id=req.model_id, valid=bool(res["valid"]), detail=res.get("detail"))


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


# =========================
# Two-phase audit endpoints
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

    pred = zk.get("prediction", float("nan"))
    if isinstance(pred, (int, float)) and (math.isnan(pred) or math.isinf(pred)):
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
