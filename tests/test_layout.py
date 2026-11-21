"""Tests for weight-layout decoding and equivalence checking."""

import base64
import itertools
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from layout import (  # noqa: E402
    LayoutError,
    check_layout,
    packed_element_count,
)
from main import app  # noqa: E402
from schemas import LayoutCheckRequest, LayoutDimensions  # noqa: E402

client = TestClient(app)


# --------------------------------------------------------------------------
# independent reference encoder (does not reuse production coordinate code)
# --------------------------------------------------------------------------

def nested_lookup(nested, order, coord):
    """Fetch a value from a nested tensor laid out along ``order``."""
    node = nested
    for axis in order:
        node = node[coord[axis]]
    return node


def reference_pack(nested, order, extents, blocks, *, pad_value=0,
                   nested_order=("oc", "cin", "ky", "kx")):
    """Encode a nested tensor into a blocked, permuted signed-INT8 buffer."""
    per_axis = []
    for axis in order:
        extent, block = extents[axis], blocks[axis]
        idx = [b * block + inner
               for b in range(-(-extent // block))
               for inner in range(block)]
        per_axis.append((axis, idx))

    canonical = ("oc", "cin", "ky", "kx")
    out = bytearray()
    for combo in itertools.product(*(idx for _, idx in per_axis)):
        coord = dict(zip((axis for axis, _ in per_axis), combo))
        if any(coord[axis] >= extents[axis] for axis in canonical):
            out.append(pad_value & 0xFF)
        else:
            out.append(nested_lookup(nested, nested_order, coord) & 0xFF)
    return bytes(out)


def b64(raw):
    return base64.b64encode(raw).decode()


def dims(oc, cin, kh, kw, ocb=1, cinb=1, kb=1):
    return {
        "output_channels": oc, "input_channels": cin,
        "kernel_height": kh, "kernel_width": kw,
        "output_channel_block": ocb, "input_channel_block": cinb,
        "kernel_block": kb,
    }


def weights_of(oc, cin, kh, kw, seed=0):
    rng = random.Random(seed)
    return [[[[rng.randint(-128, 127) for _ in range(kw)] for _ in range(kh)]
             for _ in range(cin)] for _ in range(oc)]


def feature_map_of(cin, h, w, seed=1):
    rng = random.Random(seed)
    return [[[rng.randint(0, 255) for _ in range(w)] for _ in range(h)]
            for _ in range(cin)]


def request_dict(weights, bias, fm, zps, dimensions, decoded_order,
                 packed_order, packed):
    return {
        "weights": weights,
        "bias": bias,
        "feature_map": {"height": len(fm[0]), "width": len(fm[0][0]), "data": fm},
        "input_zero_points": zps,
        "dimensions": dimensions,
        "layout": {"decoded_axis_order": decoded_order,
                   "packed_axis_order": packed_order},
        "packed_weights_base64": packed,
    }


def check(payload_dict):
    """Validate the request through the schema, then call check_layout."""
    req = LayoutCheckRequest.model_validate(payload_dict)
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


def make_case(seed=0, order=None, blocks=(1, 1, 1), decoded_order=None):
    """Build a consistent request for the given layout, with correct weights."""
    rng = random.Random(seed)
    order = order or ["kx", "oc", "cin", "ky"]
    decoded_order = decoded_order or ["oc", "cin", "ky", "kx"]
    oc, cin = rng.randint(1, 3), rng.randint(1, 3)
    kh = kw = rng.choice([1, 3])
    weights = weights_of(oc, cin, kh, kw, seed=seed)
    bias = [rng.randint(-1000, 1000) for _ in range(oc)]
    fm = feature_map_of(cin, rng.randint(1, 4), rng.randint(1, 4), seed=seed + 100)
    zps = [rng.randint(0, 255) for _ in range(cin)]
    d = dims(oc, cin, kh, kw, *blocks)
    extents = {"oc": oc, "cin": cin, "ky": kh, "kx": kw}
    blocks_map = {"oc": blocks[0], "cin": blocks[1],
                  "ky": blocks[2], "kx": blocks[2]}
    if decoded_order == ["oc", "cin", "ky", "kx"]:
        nested = weights
    else:
        assert decoded_order == ["cin", "oc", "ky", "kx"]
        nested = [[weights[o][c] for o in range(oc)] for c in range(cin)]
    raw = reference_pack(nested, order, extents, blocks_map,
                         nested_order=decoded_order)
    return request_dict(nested, bias, fm, zps, d, decoded_order, order, b64(raw))


# --------------------------------------------------------------------------
# round-trip and layout decoding
# --------------------------------------------------------------------------

@pytest.mark.parametrize("perm", list(itertools.permutations(["oc", "cin", "ky", "kx"])))
def test_all_axis_permutations_round_trip(perm):
    order = list(perm)
    res = check(make_case(seed=sum(map(len, perm)) % 97, order=order))
    assert res["matches"] is True, res["difference"]
    assert res["weights_match"] is True
    assert res["replay_match"] is True
    assert res["difference"] is None


@pytest.mark.parametrize("blocks", [(2, 2, 2), (2, 3, 2), (4, 4, 4)])
def test_blocking_round_trip_with_padding(blocks):
    res = check(make_case(seed=5, order=["ky", "kx", "cin", "oc"], blocks=blocks))
    assert res["matches"] is True, res["difference"]
    assert res["packed"]["padding_elements"] > 0
    assert (res["packed"]["padding_elements"]
            + res["packed"]["valid_elements"] == res["packed"]["bytes"])


def test_all_int8_values_survive_decode():
    weights = [[[[-128, -1, 0], [1, 127, -64], [64, -127, 32]]]]
    fm = [[[0, 255, 3], [7, 8, 9], [10, 11, 12]]]
    d = dims(1, 1, 3, 3, 2, 2, 2)
    extents = {"oc": 1, "cin": 1, "ky": 3, "kx": 3}
    blocks = {"oc": 2, "cin": 2, "ky": 2, "kx": 2}
    raw = reference_pack(weights, ["kx", "cin", "oc", "ky"], extents, blocks)
    res = check(request_dict(weights, [0], fm, [128], d,
                             ["oc", "cin", "ky", "kx"], ["kx", "cin", "oc", "ky"],
                             b64(raw)))
    assert res["matches"] is True, res["difference"]
    assert res["packed"]["valid_elements"] == 9


def test_packed_element_count():
    body = request_dict([[ [[1]] ]], [0], [[[0]]], [0],
                        dims(3, 5, 3, 3, 2, 4, 2),
                        ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                        b64(b"\x01"))
    d = LayoutDimensions.model_validate(body["dimensions"])
    assert packed_element_count(d) == 4 * 8 * 4 * 4


# --------------------------------------------------------------------------
# mismatch reporting
# --------------------------------------------------------------------------

def test_weight_mismatch_reports_first_index_and_sums():
    weights = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    bad = [[[[1, 2, 3], [4, 5, 6], [7, 8, -9]]]]
    fm = [[[2, 3, 4], [5, 6, 7], [8, 9, 10]]]
    d = dims(1, 1, 3, 3)
    extents = {"oc": 1, "cin": 1, "ky": 3, "kx": 3}
    blocks = {"oc": 1, "cin": 1, "ky": 1, "kx": 1}
    raw = reference_pack(bad, ["oc", "cin", "ky", "kx"], extents, blocks)
    res = check(request_dict(weights, [10], fm, [5], d,
                             ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                             b64(raw)))
    assert res["matches"] is False
    assert res["weight_check"]["mismatch_count"] == 1
    first = res["weight_check"]["first_mismatch"]
    assert first["index"] == {"output_channel": 0, "input_channel": 0,
                              "kernel_y": 2, "kernel_x": 2}
    assert first["reference"] == 9 and first["packed"] == -9

    diff = res["difference"]
    assert diff["kind"] == "weight_mismatch"
    assert diff["index"] == first["index"]
    # scan order first reads tap (2,2) at output (0,0); delta = 6 - 5 = 1
    assert diff["output"] == {"y": 0, "x": 0, "channel": 0}
    assert diff["step"] == 9
    assert diff["reference_partial_sum"] - diff["packed_partial_sum"] == (9 - -9) * 1


def test_weight_mismatch_not_exercised_by_feature_map():
    # Corner tap differs but a 1x1 map never reads it -> replay agrees.
    weights = [[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]
    bad = [[[[2, 1, 1], [1, 1, 1], [1, 1, 1]]]]
    fm = [[[7]]]
    d = dims(1, 1, 3, 3)
    raw = reference_pack(bad, ["oc", "cin", "ky", "kx"],
                         {"oc": 1, "cin": 1, "ky": 3, "kx": 3},
                         {"oc": 1, "cin": 1, "ky": 1, "kx": 1})
    res = check(request_dict(weights, [0], fm, [0], d,
                             ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                             b64(raw)))
    assert res["weights_match"] is False
    assert res["replay_match"] is True
    assert res["difference"]["kind"] == "weight_mismatch"
    assert res["difference"]["output"] is None
    assert res["difference"]["reference_partial_sum"] is None


def test_replay_counts_and_padding_zero_contribution():
    # 1 channel, 3x3 kernel, zp=0, 2x2 map: every output sees 4..9 taps.
    weights = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    fm = [[[9, 8], [7, 6]]]
    d = dims(1, 1, 3, 3)
    raw = reference_pack(weights, ["oc", "cin", "ky", "kx"],
                         {"oc": 1, "cin": 1, "ky": 3, "kx": 3},
                         {"oc": 1, "cin": 1, "ky": 1, "kx": 1})
    res = check(request_dict(weights, [0], fm, [0], d,
                             ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                             b64(raw)))
    assert res["replay"]["positions_checked"] == 4
    assert res["replay"]["steps_checked"] == 4 * 9
    assert res["matches"] is True


# --------------------------------------------------------------------------
# layout / payload errors -> LayoutError
# --------------------------------------------------------------------------

def test_nonzero_tail_padding_rejected():
    weights = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    d = dims(1, 1, 3, 3, 1, 1, 2)  # ky/kx blocked by 2 -> 4x4 = 16 slots
    raw = reference_pack(weights, ["oc", "cin", "ky", "kx"],
                         {"oc": 1, "cin": 1, "ky": 3, "kx": 3},
                         {"oc": 1, "cin": 1, "ky": 2, "kx": 2}, pad_value=7)
    with pytest.raises(LayoutError, match="padding"):
        check(request_dict(weights, [0], [[[0]]], [0], d,
                           ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                           b64(raw)))


def test_short_and_long_payloads_rejected():
    weights = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    d = dims(1, 1, 3, 3)
    raw = reference_pack(weights, ["oc", "cin", "ky", "kx"],
                         {"oc": 1, "cin": 1, "ky": 3, "kx": 3},
                         {"oc": 1, "cin": 1, "ky": 1, "kx": 1})
    for bad_raw in (raw[:-1], raw + b"\x00"):
        with pytest.raises(LayoutError, match="expected"):
            check(request_dict(weights, [0], [[[0]]], [0], d,
                               ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                               b64(bad_raw)))


def test_invalid_base64_rejected():
    weights = [[[[1]]]]
    with pytest.raises(LayoutError, match="base64"):
        check(request_dict(weights, [0], [[[0]]], [0], dims(1, 1, 1, 1),
                           ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                           "not base64!!"))


def test_oversized_base64_rejected_before_decode():
    weights = [[[[1]]]]
    with pytest.raises(LayoutError, match="expected bytes"):
        check(request_dict(weights, [0], [[[0]]], [0], dims(1, 1, 1, 1),
                           ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                           "A" * 100))


def test_zero_delta_hides_mismatch_from_replay():
    # Reference and packed differ but zp == activation everywhere, so the
    # replayed terms are 0 and the partial sums never split.
    weights = [[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]
    bad = [[[[2, 1, 1], [1, 1, 1], [1, 1, 1]]]]
    fm = [[[9, 9], [9, 9]]]
    d = dims(1, 1, 3, 3)
    raw = reference_pack(bad, ["oc", "cin", "ky", "kx"],
                         {"oc": 1, "cin": 1, "ky": 3, "kx": 3},
                         {"oc": 1, "cin": 1, "ky": 1, "kx": 1})
    res = check(request_dict(weights, [0], fm, [9], d,
                             ["oc", "cin", "ky", "kx"], ["oc", "cin", "ky", "kx"],
                             b64(raw)))
    assert res["weights_match"] is False
    assert res["replay_match"] is True
    assert res["difference"]["kind"] == "weight_mismatch"
    assert res["difference"]["output"] is None
    assert res["difference"]["reference_partial_sum"] is None


def test_non_canonical_decoded_order_round_trip():
    # Reference weights stored as [cin][oc][ky][kx] with matching decoded_axis_order.
    packed_req = make_case(
        seed=11, order=["kx", "ky", "cin", "oc"], blocks=(2, 3, 2),
        decoded_order=["cin", "oc", "ky", "kx"],
    )
    res = check(packed_req)
    assert res["matches"] is True, res["difference"]


# --------------------------------------------------------------------------
# API layer
# --------------------------------------------------------------------------

def test_api_ok():
    r = client.post("/v1/layout-check", json=make_case(seed=3, blocks=(2, 2, 2)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matches"] is True
    assert body["difference"] is None
    assert body["layout"]["signed_int8"] is True


def test_api_mismatch():
    p = make_case(seed=4)
    p["weights"][0][0][0][0] = (p["weights"][0][0][0][0] + 1) % 128
    r = client.post("/v1/layout-check", json=p)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matches"] is False
    assert body["weight_check"]["mismatch_count"] >= 1
    diff = body["difference"]
    assert set(diff["index"]) == {"output_channel", "input_channel",
                                  "kernel_y", "kernel_x"}
    assert diff["index"] == body["weight_check"]["first_mismatch"]["index"]
    assert "reference_partial_sum" in diff and "packed_partial_sum" in diff
    # the replay always scans the whole map, it does not stop at the first split
    h = len(p["feature_map"]["data"][0])
    w = len(p["feature_map"]["data"][0][0])
    assert body["replay"]["positions_checked"] == h * w * len(p["bias"])


def test_api_422_nonzero_padding():
    weights = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    d = dims(1, 1, 3, 3, 2, 2, 2)
    raw = reference_pack(weights, ["oc", "cin", "ky", "kx"],
                         {"oc": 1, "cin": 1, "ky": 3, "kx": 3},
                         {"oc": 2, "cin": 2, "ky": 2, "kx": 2}, pad_value=1)
    r = client.post("/v1/layout-check",
                    json=request_dict(weights, [0], [[[0]]], [0], d,
                                      ["oc", "cin", "ky", "kx"],
                                      ["oc", "cin", "ky", "kx"], b64(raw)))
    assert r.status_code == 422
    assert "padding" in r.json()["detail"].lower()


def test_api_422_short_bytes():
    p = make_case(seed=7)
    p["packed_weights_base64"] = b64(base64.b64decode(p["packed_weights_base64"])[:-1])
    r = client.post("/v1/layout-check", json=p)
    assert r.status_code == 422
    assert "expected" in r.json()["detail"].lower()


def test_api_422_invalid_base64():
    p = make_case(seed=9)
    p["packed_weights_base64"] = "***"
    r = client.post("/v1/layout-check", json=p)
    assert r.status_code == 422
    assert "base64" in r.json()["detail"].lower()


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda p: p["dimensions"].__setitem__("kernel_height", 2), "odd"),
        (lambda p: p["dimensions"].__setitem__("output_channels", 9), "output_channels"),
        (lambda p: p["dimensions"].__setitem__("input_channels", 9), "input_channels"),
        (lambda p: p["bias"].append(0), "bias"),
        (lambda p: p["input_zero_points"].__setitem__(0, 300), "uint8"),
        (lambda p: p["feature_map"].__setitem__("height", 99), "rows"),
        (lambda p: p["feature_map"]["data"][0][0].__setitem__(0, 256), "uint8"),
        (lambda p: p["layout"].__setitem__("packed_axis_order", ["oc", "oc", "ky", "kx"]), "permutation"),
        (lambda p: p["layout"].__setitem__("decoded_axis_order", ["oc", "cin", "ky", "oc"]), "permutation"),
        (lambda p: p.__setitem__("packed_weights_base64", ""), "packed_weights_base64"),
        (lambda p: p["weights"][0][0][0].__setitem__(0, 200), "INT8"),
        (lambda p: p["dimensions"].__setitem__("kernel_block", 0), "kernel_block"),
        (lambda p: p["weights"][0].__setitem__(0, []), "axis 'ky'"),
    ],
)
def test_api_validation_errors(mutate, fragment):
    p = make_case(seed=8)
    mutate(p)
    r = client.post("/v1/layout-check", json=p)
    assert r.status_code == 422, r.text
    assert fragment.lower() in str(r.json()["detail"]).lower()


def test_api_index_lists_both_endpoints():
    endpoints = client.get("/").json()["endpoints"]
    assert endpoints["analyze"] == "POST /v1/analyze"
    assert endpoints["layout_check"] == "POST /v1/layout-check"


# --------------------------------------------------------------------------
# shipped samples (README examples must stay reproducible)
# --------------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load_sample(name):
    import json

    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def test_shipped_match_sample():
    r = client.post("/v1/layout-check", json=_load_sample("layout_match.json"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matches"] is True
    assert body["difference"] is None
    assert body["packed"] == {
        "bytes": 64, "valid_elements": 36, "padding_elements": 28,
    }


def test_shipped_mismatch_sample():
    r = client.post("/v1/layout-check", json=_load_sample("layout_mismatch.json"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matches"] is False
    assert body["weight_check"] == {
        "mismatch_count": 1,
        "first_mismatch": {
            "index": {"output_channel": 0, "input_channel": 1,
                      "kernel_y": 1, "kernel_x": 2},
            "reference": 1, "packed": 2,
        },
    }
    diff = body["difference"]
    assert diff["kind"] == "weight_mismatch"
    assert diff["output"] == {"y": 0, "x": 0, "channel": 0}
    assert diff["step"] == 15
    assert (diff["reference_partial_sum"], diff["packed_partial_sum"]) == (-103, -101)
    # 102 - 100 = 2 -> terms 2*1 vs 2*2 differ by 2
    assert diff["packed_partial_sum"] - diff["reference_partial_sum"] == 2


def test_shipped_samples_regenerate_identically():
    """Re-running the sample generator must reproduce the committed files."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_layout_samples", EXAMPLES / "make_layout_samples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import itertools
    good = module.pack(module.WEIGHTS, module.PACKED_ORDER, module.BLOCKS)
    assert base64.b64decode(_load_sample("layout_match.json")["packed_weights_base64"]) == good

    bad = bytearray(good)
    pos = 0
    for combo in itertools.product(*[
        [b * module.BLOCKS[a] + inner
         for b in range(-(-module.EXTENTS[a] // module.BLOCKS[a]))
         for inner in range(module.BLOCKS[a])]
        for a in module.PACKED_ORDER
    ]):
        coord = dict(zip(module.PACKED_ORDER, combo))
        if coord == {"oc": 0, "cin": 1, "ky": 1, "kx": 2}:
            bad[pos] = (module.WEIGHTS[0][1][1][2] + 1) & 0xFF
            break
        pos += 1
    assert base64.b64decode(_load_sample("layout_mismatch.json")["packed_weights_base64"]) == bytes(bad)
