"""API-level tests via FastAPI TestClient."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from main import app  # noqa: E402

client = TestClient(app)


def valid_payload():
    return {
        "feature_map": {"height": 4, "width": 4},
        "activations": [
            {"min": 0, "max": 255, "zero_point": 128},
            {"min": 10, "max": 200, "zero_point": 100},
        ],
        "weights": [
            [[[3, -2, 1], [0, 4, -1], [2, 1, -3]], [[-1, 2, 0], [3, -3, 1], [0, 1, 2]]],
            [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[-2, 0, 2], [-2, 0, 2], [-2, 0, 2]]],
        ],
        "bias": [120, -40],
        "target_bits": 32,
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_ok():
    r = client.post("/v1/analyze", json=valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["target_ok"] is True
    assert body["overflow"] is None
    assert body["min_safe_bits"] <= 32
    assert body["stats"]["accumulation_steps"] == 2 * 3 * 3
    assert body["stats"]["output_positions"] == 16
    assert body["extremes"]["min_partial_sum"]["value"] <= 0 <= body["extremes"]["max_partial_sum"]["value"]


def test_analyze_overflow_contract():
    payload = valid_payload()
    payload["target_bits"] = 8
    r = client.post("/v1/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["target_ok"] is False
    ov = body["overflow"]
    assert ov["step"] >= 0
    assert set(ov["output"]) == {"y", "x", "channel"}
    assert ov["direction"] in ("positive", "negative")
    lo, hi = ov["accumulator_range"]
    assert lo == -(1 << 7) and hi == (1 << 7) - 1
    assert not (lo <= ov["partial_sum"] <= hi)
    # replay terms
    run = ov["bias"]
    for t in ov["terms"]:
        assert t["term"] == (t["activation"] - t["zero_point"]) * t["weight"]
        run += t["term"]
        assert t["partial_sum"] == run
    assert run == ov["partial_sum"]
    assert (ov["terms"][-1]["step"] if ov["terms"] else 0) == ov["step"]


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda p: p["weights"].__setitem__(0, p["weights"][0][:-1]), "input channels"),
        (lambda p: p["weights"][0][0].__setitem__(0, [1, 2]), "cols"),
        (lambda p: p["weights"][0][0][0].__setitem__(0, 200), "INT8"),
        (lambda p: p["weights"][0][0].__setitem__(0, [1, 2, 3, 4]), "cols"),
        (lambda p: p.__setitem__("bias", [1]), "bias"),
        (lambda p: p["activations"].__setitem__(0, {"min": 200, "max": 100, "zero_point": 0}), "min"),
        (lambda p: p["activations"].__setitem__(0, {"min": 0, "max": 300, "zero_point": 0}), "max"),
        (lambda p: p.__setitem__("target_bits", 0), "target_bits"),
    ],
)
def test_validation_errors(mutate, fragment):
    payload = valid_payload()
    mutate(payload)
    r = client.post("/v1/analyze", json=payload)
    assert r.status_code == 422
    assert fragment.lower() in str(r.json()["detail"]).lower()


def test_even_kernel_rejected():
    payload = valid_payload()
    payload["weights"] = [[[[1, 2], [3, 4]], [[1, 1], [1, 1]]]]
    payload["bias"] = [0]
    r = client.post("/v1/analyze", json=payload)
    assert r.status_code == 422
    assert "odd" in str(r.json()["detail"]).lower()
