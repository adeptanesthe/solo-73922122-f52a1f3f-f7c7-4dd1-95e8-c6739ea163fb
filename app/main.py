"""FastAPI entrypoint: INT8 conv accumulator width + weight-layout checks."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from analyzer import MAX_WORK_ITEMS, WorkTooLarge, analyze_spec
from layout import LayoutError, check_layout
from schemas import AnalyzeRequest, LayoutCheckRequest

app = FastAPI(
    title="INT8 Conv Accumulator Inspector",
    version="1.1.0",
    description=(
        "Two checks for one quantized INT8 conv layer (stride 1, odd kernel, "
        "zero-point padding, same output size). /v1/analyze reports the minimum "
        "safe signed accumulator width. /v1/layout-check decodes a device packed "
        "kernel buffer, compares it element-wise against reference weights, and "
        "replays a concrete uint8 feature map to localise the first divergence."
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


@app.post("/v1/layout-check")
def layout_check(req: LayoutCheckRequest) -> dict:
    try:
        return check_layout(
            feature_map_data=req.feature_map.data,
            feature_map_height=req.feature_map.height,
            feature_map_width=req.feature_map.width,
            zero_points=req.input_zero_points,
            reference_weights=req.weights,
            bias=req.bias,
            dimensions=req.dimensions,
            decoded_axis_order=req.layout.decoded_axis_order,
            packed_axis_order=req.layout.packed_axis_order,
            packed_base64=req.packed_weights_base64,
        )
    except LayoutError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@app.get("/")
def index() -> dict:
    return {
        "service": "int8-conv-accumulator-inspector",
        "endpoints": {
            "analyze": "POST /v1/analyze",
            "layout_check": "POST /v1/layout-check",
            "health": "GET /health",
            "docs": "GET /docs",
        },
        "limits": {"work_items": MAX_WORK_ITEMS},
    }
