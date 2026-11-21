"""Exact worst-case accumulator bit-width analysis for one quantized INT8 conv layer.

Inspection contract (fixed semantics):
  * 2-D convolution, stride 1, odd kernel, output size == input size.
  * Boundary padding uses the activation zero point of the corresponding
    channel, so a padded tap contributes exactly (zp - zp) * w == 0.
  * Accumulation order: bias is step 0, then taps are accumulated in
    (input channel, kernel row, kernel col) nesting order. Step numbers
    start at 1 for the first tap.
  * Each term is (activation - zero_point_of_its_channel) * weight.
  * Every activation cell may independently take any integer value inside
    its channel's [min, max] range.

Exactness: all taps of one output position read distinct input cells and the
cells are independent, so the min/max of the partial sum at step k equals the
sum of the per-term min/max contributions of steps 1..k, and both bounds are
simultaneously reachable. Interval arithmetic per step is therefore exact,
not conservative.
"""

from __future__ import annotations

# Guardrail for the pure-Python scan: output_channels * position_types * steps.
MAX_WORK_ITEMS = 20_000_000


class WorkTooLarge(Exception):
    """Raised when the requested layer exceeds the inspection work budget."""


def bits_for_value(v: int) -> int:
    """Minimum signed two's-complement bit width that can hold value v."""
    if v >= 0:
        return v.bit_length() + 1
    return (-v - 1).bit_length() + 1


def build_steps(cin: int, kh: int, kw: int) -> list[tuple[int, int, int]]:
    """(channel, kernel_row, kernel_col) per tap; step number = index + 1."""
    return [(c, i, j) for c in range(cin) for i in range(kh) for j in range(kw)]


def position_types(height: int, width: int, kh: int, kw: int) -> list[dict]:
    """Distinct padding masks over the output map.

    Two output positions behave identically iff they share the same set of
    in-bounds kernel rows/cols. Returns one entry per distinct mask with the
    first (scan-order) representative output position and the inclusive valid
    kernel row/col ranges.
    """
    ph, pw = kh // 2, kw // 2
    seen: dict[tuple[int, int, int, int], dict] = {}
    for y in range(height):
        i0 = max(0, ph - y)
        i1 = min(kh - 1, ph + height - 1 - y)
        for x in range(width):
            j0 = max(0, pw - x)
            j1 = min(kw - 1, pw + width - 1 - x)
            key = (i0, i1, j0, j1)
            if key not in seen:
                seen[key] = {"y": y, "x": x, "i0": i0, "i1": i1, "j0": j0, "j1": j1}
    return list(seen.values())


def _build_reproduction(
    *,
    oc: int,
    y: int,
    x: int,
    step: int,
    direction: str,
    channels: list[dict],
    weights: list,
    bias: list[int],
    height: int,
    width: int,
    kh: int,
    kw: int,
    steps: list[tuple[int, int, int]],
) -> tuple[list[dict], int]:
    """Concrete local inputs that drive the partial sum to the recorded extreme.

    Returns (terms, partial_sum) where terms covers steps 1..step in
    accumulation order. Padded taps are listed with the channel zero point and
    contribute exactly 0.
    """
    ph, pw = kh // 2, kw // 2
    terms: list[dict] = []
    run = bias[oc]
    for k in range(step):
        c, i, j = steps[k]
        ch = channels[c]
        zp = ch["zero_point"]
        wv = weights[oc][c][i][j]
        iy = y + i - ph
        ix = x + j - pw
        if 0 <= iy < height and 0 <= ix < width:
            if wv > 0:
                act = ch["max"] if direction == "positive" else ch["min"]
            elif wv < 0:
                act = ch["min"] if direction == "positive" else ch["max"]
            else:
                act = ch["min"]  # weight 0: any in-range value works, term is 0
            term = (act - zp) * wv
            terms.append(
                {
                    "step": k + 1,
                    "padded": False,
                    "input": {"y": iy, "x": ix, "channel": c},
                    "activation": act,
                    "zero_point": zp,
                    "weight": wv,
                    "term": term,
                    "partial_sum": run + term,
                }
            )
        else:
            terms.append(
                {
                    "step": k + 1,
                    "padded": True,
                    "input": None,
                    "activation": zp,
                    "zero_point": zp,
                    "weight": wv,
                    "term": 0,
                    "partial_sum": run,
                }
            )
        run += terms[-1]["term"]
    return terms, run


def analyze_spec(
    *,
    height: int,
    width: int,
    channels: list[dict],
    weights: list,
    bias: list[int],
    target_bits: int,
) -> dict:
    """Run the full inspection. All inputs are assumed schema-validated."""
    cin = len(channels)
    cout = len(weights)
    kh = len(weights[0][0])
    kw = len(weights[0][0][0])
    steps = build_steps(cin, kh, kw)
    n_steps = len(steps)

    d_lo = [ch["min"] - ch["zero_point"] for ch in channels]
    d_hi = [ch["max"] - ch["zero_point"] for ch in channels]

    types = position_types(height, width, kh, kw)
    for t in types:
        t["valid"] = [
            k
            for k, (_c, i, j) in enumerate(steps)
            if t["i0"] <= i <= t["i1"] and t["j0"] <= j <= t["j1"]
        ]

    work_items = cout * sum(len(t["valid"]) + 1 for t in types)
    if work_items > MAX_WORK_ITEMS:
        raise WorkTooLarge(
            f"inspection would take ~{work_items} accumulate/check iterations "
            f"(limit {MAX_WORK_ITEMS}); reduce channels, kernel or output channels"
        )

    tgt_lo = -(1 << (target_bits - 1))
    tgt_hi = (1 << (target_bits - 1)) - 1

    min_safe_bits = 1
    worst_min: dict | None = None
    worst_max: dict | None = None
    overflow: dict | None = None  # earliest target-range violation

    def observe(lo: int, hi: int, step_no: int, y: int, x: int, oc: int) -> None:
        nonlocal min_safe_bits, worst_min, worst_max, overflow
        need = max(bits_for_value(lo), bits_for_value(hi))
        if need > min_safe_bits:
            min_safe_bits = need
        if worst_min is None or lo < worst_min["value"]:
            worst_min = {
                "value": lo,
                "step": step_no,
                "output": {"y": y, "x": x, "channel": oc},
            }
        if worst_max is None or hi > worst_max["value"]:
            worst_max = {
                "value": hi,
                "step": step_no,
                "output": {"y": y, "x": x, "channel": oc},
            }
        for direction, value, excess in (
            ("negative", lo, tgt_lo - lo),
            ("positive", hi, hi - tgt_hi),
        ):
            if excess > 0:
                cand = {
                    "step": step_no,
                    "y": y,
                    "x": x,
                    "oc": oc,
                    "direction": direction,
                    "value": value,
                    "excess": excess,
                }
                # earliest step wins; tie-break on larger overshoot, then scan order
                if (
                    overflow is None
                    or step_no < overflow["step"]
                    or (step_no == overflow["step"] and excess > overflow["excess"])
                ):
                    overflow = cand

    for oc in range(cout):
        w_oc = weights[oc]
        term_min = [0] * n_steps
        term_max = [0] * n_steps
        for k, (c, i, j) in enumerate(steps):
            wv = w_oc[c][i][j]
            a = d_lo[c] * wv
            b = d_hi[c] * wv
            term_min[k] = a if a < b else b
            term_max[k] = b if a < b else a
        b0 = bias[oc]
        for t in types:
            lo = hi = b0
            observe(lo, hi, 0, t["y"], t["x"], oc)  # step 0: bias only
            for k in t["valid"]:
                lo += term_min[k]
                hi += term_max[k]
                observe(lo, hi, k + 1, t["y"], t["x"], oc)

    result = {
        "min_safe_bits": min_safe_bits,
        "target_bits": target_bits,
        "target_ok": overflow is None,
        "signed_range_for_target": [tgt_lo, tgt_hi],
        "stats": {
            "accumulation_steps": n_steps,
            "output_positions": height * width,
            "position_types": len(types),
            "output_channels": cout,
        },
        "extremes": {
            "min_partial_sum": worst_min,
            "max_partial_sum": worst_max,
        },
        "overflow": None,
    }

    if overflow is not None:
        terms, partial_sum = _build_reproduction(
            oc=overflow["oc"],
            y=overflow["y"],
            x=overflow["x"],
            step=overflow["step"],
            direction=overflow["direction"],
            channels=channels,
            weights=weights,
            bias=bias,
            height=height,
            width=width,
            kh=kh,
            kw=kw,
            steps=steps,
        )
        assert partial_sum == overflow["value"], "reproduction mismatch"
        limit = tgt_hi if overflow["direction"] == "positive" else tgt_lo
        result["overflow"] = {
            "step": overflow["step"],
            "output": {"y": overflow["y"], "x": overflow["x"], "channel": overflow["oc"]},
            "direction": overflow["direction"],
            "accumulator_range": [tgt_lo, tgt_hi],
            "partial_sum": partial_sum,
            "exceeds_by": overflow["excess"],
            "limit_exceeded": limit,
            "bias": bias[overflow["oc"]],
            "terms": terms,
            "note": (
                "Steps 1..N in accumulation order (input channel, kernel row, "
                "kernel col); step 0 is the bias. 'padded' entries read the "
                "channel zero point and contribute exactly 0. Replaying "
                "bias + terms reproduces partial_sum at the reported step."
            ),
        }
    return result
