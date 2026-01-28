import base64
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_ENV = os.getenv("APP_ENV", "dev")

MODELS = [
    {
        "model_id": "risk_model_v1",
        "description": "Toy risk score model (PoC)",
        "model_hash": "sha256:demo-model-hash-v1",
    }
]

app = FastAPI(title="Decision Integrity Proof Server", version="0.1.0")


class ProveRequest(BaseModel):
    model_id: str = Field(..., description="Which model version to use")
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

class CommitRequest(BaseModel):
    model_id: str
    prediction: float
    nonce: str
    context: Optional[Dict[str, Any]] = None


class CommitResponse(BaseModel):
    commitment_hash: str


@app.get("/health")
def health():
    return {"ok": True, "env": APP_ENV}


@app.get("/models")
def list_models():
    return [
        {
            "model_id": m["model_id"],
            "description": m["description"],
            "model_hash": m["model_hash"],
        }
        for m in MODELS
    ]


def model_predict_stub(features: List[float]) -> float:
    weights = [0.4, -0.2, 0.1, 0.05]
    s = 0.0
    for i, x in enumerate(features[: len(weights)]):
        s += weights[i] * x
    return max(0.0, min(1.0, 0.5 + s))


def fake_proof(features: List[float], model_id: str) -> Dict[str, str]:
    payload = json.dumps({"features": features, "model_id": model_id}, sort_keys=True).encode()
    proof = hashlib.sha256(payload).digest()
    public_inputs = hashlib.sha256(b"public_inputs|" + payload).digest()
    return {
        "proof_b64": base64.b64encode(proof).decode(),
        "public_inputs_b64": base64.b64encode(public_inputs).decode(),
    }


@app.post("/prove", response_model=ProveResponse)
def prove(req: ProveRequest):
    # Example "policy constraints" for PoC
    if len(req.features) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 features for this PoC.")
    if any((x < -10.0 or x > 10.0) for x in req.features):
        raise HTTPException(status_code=400, detail="Feature out of allowed range (-10..10).")

    model = next((m for m in MODELS if m["model_id"] == req.model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Unknown model_id")

    prediction = model_predict_stub(req.features)
    zk = fake_proof(req.features, req.model_id)

    return ProveResponse(
        model_id=req.model_id,
        model_hash=model["model_hash"],
        prediction=prediction,
        proof_b64=zk["proof_b64"],
        public_inputs_b64=zk["public_inputs_b64"],
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    # PoC v1: always valid. Later we'll plug real EZKL verification here.
    return VerifyResponse(model_id=req.model_id, valid=True)
