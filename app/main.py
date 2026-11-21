"""FastAPI entrypoint: single-layer INT8 conv accumulator width inspector."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from analyzer import MAX_WORK_ITEMS, WorkTooLarge, analyze_spec
from schemas import AnalyzeRequest

app = FastAPI(
    title="INT8 Conv Accumulator Width Inspector",
    version="1.0.0",
    description=(
        "Checks one quantized INT8 conv layer (stride 1, odd kernel, zero-point "
        "padding, same output size) for the minimum signed accumulator bit "
        "width that is safe at every output position and every accumulation "
        "step. If the target width is insufficient, returns the earliest "
        "possible overflow with output coordinates and the local inputs that "
        "reproduce it."
    ),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    channels = [ch.model_dump() for ch in req.activations]
    try:
        return analyze_spec(
            height=req.feature_map.height,
            width=req.feature_map.width,
            channels=channels,
            weights=req.weights,
            bias=req.bias,
            target_bits=req.target_bits,
        )
    except WorkTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@app.get("/")
def index() -> dict:
    return {
        "service": "int8-conv-accumulator-inspector",
        "endpoints": {"analyze": "POST /v1/analyze", "health": "GET /health", "docs": "GET /docs"},
        "limits": {"work_items": MAX_WORK_ITEMS},
    }
