"""Unit tests for the analyzer, including exhaustive brute-force cross-checks."""

import itertools
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from analyzer import (  # noqa: E402
    WorkTooLarge,
    analyze_spec,
    bits_for_value,
    build_steps,
    position_types,
)


# --------------------------------------------------------------------------
# brute-force reference: enumerate every activation assignment explicitly
# --------------------------------------------------------------------------

def brute_force(height, width, channels, weights, bias, target_bits):
    """Exhaustive ground truth.

    Returns (min_safe_bits, earliest_overflow_step_or_None, global_min, global_max).
    """
    cin = len(channels)
    cout = len(weights)
    kh = len(weights[0][0])
    kw = len(weights[0][0][0])
    ph, pw = kh // 2, kw // 2
    tgt_lo = -(1 << (target_bits - 1))
    tgt_hi = (1 << (target_bits - 1)) - 1

    cells = [(y, x, c) for y in range(height) for x in range(width) for c in range(cin)]
    domains = [range(channels[c]["min"], channels[c]["max"] + 1) for (_, _, c) in cells]

    min_safe = 1
    earliest = None
    gmin = None
    gmax = None

    def observe(acc, step):
        nonlocal min_safe, earliest, gmin, gmax
        min_safe = max(min_safe, bits_for_value(acc))
        gmin = acc if gmin is None else min(gmin, acc)
        gmax = acc if gmax is None else max(gmax, acc)
        if (acc < tgt_lo or acc > tgt_hi) and (earliest is None or step < earliest):
            earliest = step

    for combo in itertools.product(*domains):
        val = dict(zip(cells, combo))
        for oc in range(cout):
            for y in range(height):
                for x in range(width):
                    acc = bias[oc]
                    observe(acc, 0)
                    step = 0
                    for c in range(cin):
                        zp = channels[c]["zero_point"]
                        for i in range(kh):
                            for j in range(kw):
                                step += 1
                                iy, ix = y + i - ph, x + j - pw
                                a = val[(iy, ix, c)] if (0 <= iy < height and 0 <= ix < width) else zp
                                acc += (a - zp) * weights[oc][c][i][j]
                                observe(acc, step)
    return min_safe, earliest, gmin, gmax


def random_spec(rng):
    height = rng.randint(1, 2)
    width = rng.randint(1, 2)
    cin = rng.randint(1, 2)
    cout = rng.randint(1, 2)
    k = rng.choice([1, 3])
    channels = []
    for _ in range(cin):
        lo = rng.randint(0, 2)
        hi = lo + rng.randint(0, 1)  # tiny domains keep enumeration cheap
        zp = rng.randint(0, 4)  # deliberately sometimes outside [lo, hi]
        channels.append({"min": lo, "max": hi, "zero_point": zp})
    weights = [
        [[[rng.randint(-3, 3) for _ in range(k)] for _ in range(k)] for _ in range(cin)]
        for _ in range(cout)
    ]
    bias = [rng.randint(-5, 5) for _ in range(cout)]
    target_bits = rng.randint(3, 9)
    return height, width, channels, weights, bias, target_bits


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,bits",
    [(0, 1), (1, 2), (-1, 1), (-2, 2), (127, 8), (128, 9), (-128, 8), (-129, 9),
     (2**31 - 1, 32), (-(2**31), 32), (2**31, 33)],
)
def test_bits_for_value(value, bits):
    assert bits_for_value(value) == bits


def test_step_order_channel_row_col():
    assert build_steps(2, 1, 2) == [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]
    assert build_steps(1, 2, 2) == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]


def test_position_types_interior_and_corners():
    types = position_types(4, 4, 3, 3)
    assert len(types) == 9  # 3 row-masks x 3 col-masks
    interior = [t for t in types if (t["i0"], t["i1"], t["j0"], t["j1"]) == (0, 2, 0, 2)]
    assert interior and (interior[0]["y"], interior[0]["x"]) == (1, 1)
    corner = [t for t in types if (t["y"], t["x"]) == (0, 0)][0]
    assert (corner["i0"], corner["j0"]) == (1, 1)  # top row / left col are padding
    assert len(position_types(1, 1, 3, 3)) == 1


# --------------------------------------------------------------------------
# hand-computed cases
# --------------------------------------------------------------------------

def test_hand_computed_1x1():
    # 1x1 map, 1 channel, 1x1 kernel: acc = bias + (a - zp) * w, a in [0, 255]
    res = analyze_spec(
        height=1, width=1,
        channels=[{"min": 0, "max": 255, "zero_point": 0}],
        weights=[[[[2]]]],
        bias=[0],
        target_bits=10,
    )
    assert res["min_safe_bits"] == 10  # max partial sum = 510
    assert res["target_ok"] is True
    assert res["overflow"] is None
    assert res["extremes"]["max_partial_sum"]["value"] == 510
    assert res["extremes"]["max_partial_sum"]["step"] == 1


def test_hand_computed_overflow_step_and_value():
    res = analyze_spec(
        height=1, width=1,
        channels=[{"min": 0, "max": 255, "zero_point": 0}],
        weights=[[[[2]]]],
        bias=[0],
        target_bits=9,  # signed 9-bit max is 255 < 510
    )
    assert res["target_ok"] is False
    ov = res["overflow"]
    assert ov["step"] == 1
    assert ov["direction"] == "positive"
    assert ov["partial_sum"] == 510
    assert ov["output"] == {"y": 0, "x": 0, "channel": 0}
    assert ov["terms"][0]["activation"] == 255


def test_bias_is_step_zero_and_can_overflow():
    res = analyze_spec(
        height=2, width=2,
        channels=[{"min": 0, "max": 3, "zero_point": 1}],
        weights=[[[[1]]]],
        bias=[1000],
        target_bits=8,
    )
    ov = res["overflow"]
    assert ov["step"] == 0
    assert ov["terms"] == []
    assert ov["partial_sum"] == 1000


def test_padding_contributes_exactly_zero():
    # 2x2 map, 3x3 kernel of weight 1, zp=0: corner sees 4 taps, rest padding.
    res = analyze_spec(
        height=2, width=2,
        channels=[{"min": 0, "max": 10, "zero_point": 0}],
        weights=[[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]],
        bias=[0],
        target_bits=8,
    )
    # global max = full 3x3 window is impossible on a 2x2 map; best is a corner
    # with 4 in-bounds taps -> 40. But the center-window bound would be 90.
    assert res["extremes"]["max_partial_sum"]["value"] == 40
    assert res["min_safe_bits"] == bits_for_value(40)


def test_zero_point_outside_range():
    # ch0: zp below [min, max] -> (a - zp) always positive
    # ch1: zp above [min, max] -> (a - zp) always negative
    res = analyze_spec(
        height=1, width=1,
        channels=[
            {"min": 10, "max": 20, "zero_point": 3},
            {"min": 10, "max": 20, "zero_point": 25},
        ],
        weights=[[[[2]], [[2]]]],  # 1 out channel, 2 in channels, 1x1 kernel
        bias=[-50],
        target_bits=8,
    )
    # step0 = -50; step1 = -50 + [14, 34]; step2 = step1 + [-30, -10]
    assert res["extremes"]["min_partial_sum"]["value"] == -50 + 14 - 30
    assert res["extremes"]["min_partial_sum"]["step"] == 2
    assert res["extremes"]["max_partial_sum"]["value"] == -50 + 34
    assert res["extremes"]["max_partial_sum"]["step"] == 1


# --------------------------------------------------------------------------
# exhaustive cross-check
# --------------------------------------------------------------------------

def test_bruteforce_crosscheck_random_specs():
    rng = random.Random(20261120)
    for _ in range(60):
        height, width, channels, weights, bias, target_bits = random_spec(rng)
        res = analyze_spec(
            height=height, width=width, channels=channels,
            weights=weights, bias=bias, target_bits=target_bits,
        )
        exp_bits, exp_step, gmin, gmax = brute_force(
            height, width, channels, weights, bias, target_bits
        )
        assert res["min_safe_bits"] == exp_bits
        assert res["extremes"]["min_partial_sum"]["value"] == gmin
        assert res["extremes"]["max_partial_sum"]["value"] == gmax
        if exp_step is None:
            assert res["target_ok"] is True
            assert res["overflow"] is None
        else:
            assert res["target_ok"] is False
            assert res["overflow"]["step"] == exp_step


def test_reproduction_replays_to_overflow():
    rng = random.Random(7)
    checked = 0
    while checked < 25:
        height, width, channels, weights, bias, target_bits = random_spec(rng)
        res = analyze_spec(
            height=height, width=width, channels=channels,
            weights=weights, bias=bias, target_bits=target_bits,
        )
        if res["overflow"] is None or res["overflow"]["step"] == 0:
            continue
        checked += 1
        ov = res["overflow"]
        lo, hi = ov["accumulator_range"]
        assert not (lo <= ov["partial_sum"] <= hi)

        # replay: bias + listed terms must reproduce every running partial sum
        run = ov["bias"]
        assert run == bias[ov["output"]["channel"]]
        last_step = 0
        for t in ov["terms"]:
            assert t["step"] == last_step + 1
            last_step = t["step"]
            ch = channels[t["input"]["channel"]] if not t["padded"] else None
            if t["padded"]:
                assert t["input"] is None
                assert t["activation"] == t["zero_point"]
                assert t["term"] == 0
            else:
                assert ch["min"] <= t["activation"] <= ch["max"]
                assert t["term"] == (t["activation"] - t["zero_point"]) * t["weight"]
            run += t["term"]
            assert t["partial_sum"] == run
        assert last_step == ov["step"]
        assert run == ov["partial_sum"]

        # listed in-bounds inputs must lie inside the kernel window of the output
        ph = len(weights[0][0]) // 2
        pw = len(weights[0][0][0]) // 2
        for t in ov["terms"]:
            if t["padded"]:
                continue
            dy = t["input"]["y"] - ov["output"]["y"]
            dx = t["input"]["x"] - ov["output"]["x"]
            assert -ph <= dy <= ph and -pw <= dx <= pw


def test_work_guard(monkeypatch):
    import analyzer

    monkeypatch.setattr(analyzer, "MAX_WORK_ITEMS", 10)
    with pytest.raises(WorkTooLarge):
        analyze_spec(
            height=4, width=4,
            channels=[{"min": 0, "max": 255, "zero_point": 128}] * 3,
            weights=[[[[1, 1, 1], [1, 1, 1], [1, 1, 1]] for _ in range(3)]],
            bias=[0],
            target_bits=32,
        )
