import base64
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

APP_ENV = os.getenv("APP_ENV", "dev")
ADMIN_KEY = os.getenv("ADMIN_KEY", "dev-admin-key")

# Demo toggle: when enabled, server will intentionally corrupt proofs to show verification failures
TAMPER_PROOF = False

# Mutable in-memory model registry for PoC (admin endpoint can rotate model_hash)
# SAFE FRAMING: "Geopolitical escalation risk forecast (next 7 days)" using abstracted public indicators.
MODELS = [
    {
        "model_id": "geo_escalation_7d_v1",
        "description": "Geopolitical escalation risk forecast (next 7 days) — PoC",
        "model_hash": "sha256:geo-escalation-7d-demo-v1",
        # Optional: purely descriptive, not used by computation
        "feature_schema": [
            "news_escalation_intensity",
            "diplomatic_tension_index",
            "event_activity_index",
            "uncertainty_score",
        ],
        "policy_note": "Demo uses abstracted public indicators only. Not operational intelligence.",
    }
]

app = FastAPI(title="Decision Integrity Proof Server", version="0.3.0")


# =========================
# Request/Response Models
# =========================

class ProveRequest(BaseModel):
    model_id: str = Field(..., description="Which approved model version to use")
    features: List[float] = Field(..., description="Numeric feature vector (PoC)")
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
    # Keep constraints simple + explicit for PoC
    if len(features) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 features for this PoC.")
    if any((x < -10.0 or x > 10.0) for x in features):
        raise HTTPException(status_code=400, detail="Feature out of allowed range (-10..10).")


def predict_geo_escalation_stub(features: List[float]) -> float:
    """
    Deterministic toy forecaster for PoC:
      features[0] = news_escalation_intensity
      features[1] = diplomatic_tension_index
      features[2] = event_activity_index
      features[3] = uncertainty_score  (higher uncertainty reduces confidence)
    Output: probability-like score in [0,1]
    """
    # Weights chosen only for a stable demo behavior, not for real forecasting
    w = [0.55, 0.35, 0.45, -0.25]
    s = 0.0
    for i in range(min(len(w), len(features))):
        s += w[i] * features[i]

    # Center around 0.5 for reasonable looking demos
    p = 0.5 + 0.2 * s
    return max(0.0, min(1.0, p))


def features_hash_hex(features: List[float]) -> str:
    payload = json.dumps(features, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def fake_proof_v2(model_hash: str, features: List[float]) -> Dict[str, str]:
    """
    PoC proof scheme with real pass/fail verification:

      public_inputs = sha256(features)  (safe fingerprint of inputs)
      proof         = sha256(model_hash || public_inputs_bytes)

    verify() recomputes and compares.
    """
    feat_payload = json.dumps(features, separators=(",", ":"), sort_keys=False).encode()
    fh_bytes = hashlib.sha256(feat_payload).digest()

    public_inputs_b64 = base64.b64encode(fh_bytes).decode()
    proof_bytes = hashlib.sha256(model_hash.encode() + fh_bytes).digest()
    proof_b64 = base64.b64encode(proof_bytes).decode()

    return {"proof_b64": proof_b64, "public_inputs_b64": public_inputs_b64}


def maybe_tamper_proof(proof_b64: str) -> str:
    """
    Intentionally corrupts the proof for demo purposes.
    Keeps base64 length same and still base64-ish to avoid decode errors.
    """
    if not proof_b64:
        return proof_b64
    first = "A" if proof_b64[0] != "A" else "B"
    return first + proof_b64[1:]


def compute_commitment(model_hash: str, feat_hash_hex: str, nonce: str, context: Dict[str, Any]) -> str:
    payload = {
        "model_hash": model_hash,
        "features_hash": feat_hash_hex,
        "nonce": nonce,
        "context": context or {},
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# =========================
# Basic endpoints
# =========================

@app.get("/health")
def health():
    return {"ok": True, "env": APP_ENV}


@app.get("/version")
def version():
    return {"version": "0.3.0-geo-escalation-demo"}


@app.get("/models")
def list_models():
    # Include metadata useful for UI labeling
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

    # PoC: model selection (only one model right now)
    prediction = predict_geo_escalation_stub(req.features)

    zk = fake_proof_v2(model["model_hash"], req.features)
    if TAMPER_PROOF:
        zk["proof_b64"] = maybe_tamper_proof(zk["proof_b64"])

    return ProveResponse(
        model_id=req.model_id,
        model_hash=model["model_hash"],
        prediction=prediction,
        proof_b64=zk["proof_b64"],
        public_inputs_b64=zk["public_inputs_b64"],
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    model = get_model(req.model_id)

    try:
        fh_bytes = base64.b64decode(req.public_inputs_b64.encode())
        provided_proof = base64.b64decode(req.proof_b64.encode())
    except Exception:
        return VerifyResponse(model_id=req.model_id, valid=False)

    expected = hashlib.sha256(model["model_hash"].encode() + fh_bytes).digest()
    return VerifyResponse(model_id=req.model_id, valid=(provided_proof == expected))


# =========================
# Commitment endpoints
# =========================

@app.post("/commit", response_model=CommitResponse)
def commit(req: CommitRequest):
    """
    Commitment over (model_hash, prediction, nonce, context).
    Useful for some demos, but commit_case/reveal is the stronger anti-backfill workflow.
    """
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
    """
    Phase 1: lock inputs + model version + policy context BEFORE outcome is known.
    Returns a commitment hash and a features hash.
    """
    model = get_model(req.model_id)
    enforce_policy_constraints(req.features)

    fh = features_hash_hex(req.features)
    commitment = compute_commitment(model["model_hash"], fh, req.nonce, req.context or {})
    return CommitCaseResponse(commitment_hash=commitment, features_hash=fh)


@app.post("/reveal", response_model=RevealResponse)
def reveal(req: RevealRequest):
    """
    Phase 2: generate outcome + proof and verify it matches the previously recorded commitment.
    """
    model = get_model(req.model_id)
    enforce_policy_constraints(req.features)

    fh = features_hash_hex(req.features)
    commitment = compute_commitment(model["model_hash"], fh, req.nonce, req.context or {})

    prediction = predict_geo_escalation_stub(req.features)

    zk = fake_proof_v2(model["model_hash"], req.features)
    if TAMPER_PROOF:
        zk["proof_b64"] = maybe_tamper_proof(zk["proof_b64"])

    matches = (commitment == req.expected_commitment_hash)

    return RevealResponse(
        model_id=req.model_id,
        model_hash=model["model_hash"],
        prediction=prediction,
        proof_b64=zk["proof_b64"],
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
