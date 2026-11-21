#!/usr/bin/env python3
"""Generate the paired sample requests for POST /v1/layout-check.

Writes:
  examples/layout_match.json      - device buffer matches the reference
  examples/layout_mismatch.json   - one packed weight is corrupted

The packed buffer is built from packed_axis_order = ["kx","ky","cin","oc"]
with blocks (2, 2, 2); the reference weights are canonical [oc][cin][ky][kx].
Only the standard library is used.
"""

from __future__ import annotations

import base64
import itertools
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# canonical [oc][cin][ky][kx], 2 out channels, 2 in channels, 3x3 kernel
WEIGHTS = [
    [
        [[3, -2, 1], [0, 4, -1], [2, 1, -3]],
        [[-1, 2, 0], [3, -3, 1], [0, 1, 2]],
    ],
    [
        [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
        [[-2, 0, 2], [-2, 1, 2], [-2, 0, 2]],
    ],
]

BIAS = [120, -40]
HEIGHT = WIDTH = 4
CIN = 2
DIMENSIONS = {
    "output_channels": 2,
    "input_channels": CIN,
    "kernel_height": 3,
    "kernel_width": 3,
    "output_channel_block": 2,
    "input_channel_block": 2,
    "kernel_block": 2,
}
DECODED_ORDER = ["oc", "cin", "ky", "kx"]
PACKED_ORDER = ["kx", "ky", "cin", "oc"]
BLOCKS = {"oc": 2, "cin": 2, "ky": 2, "kx": 2}
EXTENTS = {"oc": 2, "cin": CIN, "ky": 3, "kx": 3}


def feature_map() -> list[list[list[int]]]:
    """Deterministic uint8 map [cin][y][x]; distinct per channel and cell."""
    planes = []
    for c in range(CIN):
        plane = [
            [((c + 1) * 37 + y * 61 + x * 17 + 11) % 256 for x in range(WIDTH)]
            for y in range(HEIGHT)
        ]
        planes.append(plane)
    return planes


def pack(nested, order, block_sizes, pad_value=0) -> bytes:
    """Encode a canonical tensor into the blocked, permuted signed-INT8 buffer."""
    idx_per_axis = []
    for axis in order:
        extent, block = EXTENTS[axis], block_sizes[axis]
        idx_per_axis.append(
            [b * block + inner
             for b in range(-(-extent // block))
             for inner in range(block)]
        )
    out = bytearray()
    for combo in itertools.product(*idx_per_axis):
        coord = dict(zip(order, combo))
        if any(coord[a] >= EXTENTS[a] for a in EXTENTS):
            out.append(pad_value & 0xFF)
        else:
            value = nested[coord["oc"]][coord["cin"]][coord["ky"]][coord["kx"]]
            out.append(value & 0xFF)
    return bytes(out)


def request(packed: bytes) -> dict:
    return {
        "weights": WEIGHTS,
        "bias": BIAS,
        "feature_map": {
            "height": HEIGHT,
            "width": WIDTH,
            "data": feature_map(),
        },
        "input_zero_points": [128, 100],
        "dimensions": DIMENSIONS,
        "layout": {
            "decoded_axis_order": DECODED_ORDER,
            "packed_axis_order": PACKED_ORDER,
        },
        "packed_weights_base64": base64.b64encode(packed).decode(),
    }


def main() -> None:
    good = pack(WEIGHTS, PACKED_ORDER, BLOCKS)

    # Corrupt oc=0, cin=1, ky=1, kx=2 (value 1 -> 2) inside the device buffer.
    bad_coord = {"oc": 0, "cin": 1, "ky": 1, "kx": 2}
    expected = WEIGHTS[0][1][1][2]
    bad = bytearray(good)
    pos = 0
    for combo in itertools.product(*[
        [b * BLOCKS[a] + inner
         for b in range(-(-EXTENTS[a] // BLOCKS[a]))
         for inner in range(BLOCKS[a])]
        for a in PACKED_ORDER
    ]):
        coord = dict(zip(PACKED_ORDER, combo))
        if coord == bad_coord:
            assert bad[pos] == expected & 0xFF
            bad[pos] = (expected + 1) & 0xFF
            break
        pos += 1
    else:  # pragma: no cover
        raise SystemExit("target weight not found in packed buffer")

    (HERE / "layout_match.json").write_text(
        json.dumps(request(good), indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "layout_mismatch.json").write_text(
        json.dumps(request(bytes(bad)), indent=2) + "\n", encoding="utf-8"
    )
    print(f"packed bytes: {len(good)}")
    print("wrote layout_match.json and layout_mismatch.json")


if __name__ == "__main__":
    main()
