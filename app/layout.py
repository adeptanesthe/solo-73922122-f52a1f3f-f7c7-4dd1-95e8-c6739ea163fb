"""Weight-layout equivalence checking for on-device INT8 convolution kernels.

The device may upload its packed kernel buffer in a blocked, permuted layout.
This module decodes that buffer back to the canonical
``[out_channel][in_channel][ky][kx]`` tensor and verifies it two ways:

1. Element-wise comparison of every *effective* (non-padding) weight against
   the reference kernel supplied by the deployer, in canonical
   ``(oc, cin, ky, kx)`` order. The first mismatch is reported.
2. Step-by-step replay of the convolution over one concrete uint8 feature map,
   using the same semantics as :mod:`analyzer`: stride 1, odd kernel, output
   size == input size, boundary padding with the per-channel activation zero
   point. The reference weights and the layout-decoded weights are accumulated
   side by side. The replay locates the first output position that reads the
   mismatching tap (its output coordinate and both partial sums) and, as a
   safety net, also reports any later partial-sum divergence.

The headline ``difference`` therefore follows the order of the check: the
element-wise mismatch leads, and the replay supplies its output coordinate and
the two partial sums. If the supplied feature map never alters that tap (for
example, its delta is 0, or the map is too small to read it), the coordinates
are null while ``weight_check`` still reports the bad index.

Packed buffer contract
----------------------
``packed_axis_order`` is a permutation of ``oc``, ``cin``, ``ky``, ``kx``.
For every axis the buffer stores ``ceil(extent / block)`` blocks of ``block``
entries, with the block index iterated before the in-block index.  Unblocked
axes simply have ``block == 1``.  Padding therefore only ever occurs in the
tail of the final block of an axis; every padding byte must decode to 0.
Values are signed INT8 (``0x80..0xFF`` decode as ``-128..-1``).
"""

from __future__ import annotations

import base64
import binascii
from typing import Iterator

import analyzer

AXES = ("oc", "cin", "ky", "kx")

MAX_PACKED_ELEMENTS = 4_000_000


class LayoutError(ValueError):
    """Invalid layout, packed payload shape/content, or feature map."""


def axis_index(coord: dict[str, int]) -> dict[str, int]:
    """Canonical index dictionary for a packed coordinate."""
    return {
        "output_channel": coord["oc"],
        "input_channel": coord["cin"],
        "kernel_y": coord["ky"],
        "kernel_x": coord["kx"],
    }


# --------------------------------------------------------------------------
# layout order helpers
# --------------------------------------------------------------------------

def check_axis_order(order: list[str], field: str) -> list[str]:
    if len(order) != 4 or set(order) != set(AXES):
        raise LayoutError(
            f"{field} must be a permutation of {list(AXES)}, got {order}"
        )
    return order


def axis_extents(dimensions) -> dict[str, int]:
    return {
        "oc": dimensions.output_channels,
        "cin": dimensions.input_channels,
        "ky": dimensions.kernel_height,
        "kx": dimensions.kernel_width,
    }


def axis_blocks(dimensions) -> dict[str, int]:
    return {
        "oc": dimensions.output_channel_block,
        "cin": dimensions.input_channel_block,
        "ky": dimensions.kernel_block,
        "kx": dimensions.kernel_block,
    }


def padded_extent(extent: int, block: int) -> int:
    return -(-extent // block) * block


def packed_element_count(dimensions) -> int:
    total = 1
    extents = axis_extents(dimensions)
    blocks = axis_blocks(dimensions)
    for axis in AXES:
        total *= padded_extent(extents[axis], blocks[axis])
    return total


def iter_packed_coords(
    order: list[str], extents: dict[str, int], blocks: dict[str, int]
) -> Iterator[dict[str, int]]:
    """Yield every packed slot in buffer order (block index before in-block)."""
    spec = [(axis, extents[axis], blocks[axis]) for axis in order]

    def walk(pos: int, coord: dict[str, int]) -> Iterator[dict[str, int]]:
        if pos == len(spec):
            yield dict(coord)
            return
        axis, extent, block = spec[pos]
        n_blocks = -(-extent // block)
        for block_idx in range(n_blocks):
            for inner in range(block):
                coord[axis] = block_idx * block + inner
                yield from walk(pos + 1, coord)

    yield from walk(0, {})


# --------------------------------------------------------------------------
# decoded tensor (reference and packed) handling
# --------------------------------------------------------------------------

def decode_nested(
    nested, order: list[str], extents: dict[str, int]
) -> dict[tuple[int, int, int, int], int]:
    """Map a nested tensor in ``order`` axis order to canonical coordinates."""
    out: dict[tuple[int, int, int, int], int] = {}

    def walk(node, depth: int, coord: dict[str, int]) -> None:
        if depth == 4:
            raise LayoutError("decoded weights are deeper than 4 axes")
        axis = order[depth]
        want = extents[axis]
        if not isinstance(node, list):
            raise LayoutError(f"decoded weights axis '{axis}' must be a list")
        if len(node) != want:
            raise LayoutError(
                f"decoded weights axis '{axis}' has length {len(node)}, expected {want}"
            )
        if depth == 3:
            for i, v in enumerate(node):
                if not isinstance(v, int) or not -128 <= v <= 127:
                    raise LayoutError(
                        f"decoded weight at axis '{axis}' index {i} is "
                        f"{v!r}, outside signed INT8 [-128, 127]"
                    )
                coord[axis] = i
                out[(coord["oc"], coord["cin"], coord["ky"], coord["kx"])] = v
            return
        for i, child in enumerate(node):
            coord[axis] = i
            walk(child, depth + 1, coord)

    walk(nested, 0, {})
    return out


def unpack_packed(
    packed_b64: str,
    order: list[str],
    dimensions,
) -> tuple[dict[tuple[int, int, int, int], int], int, int]:
    """Decode the device buffer.

    Returns ``(weights, valid_elements, packed_bytes)``.  Raises
    :class:`LayoutError` for bad base64, wrong byte count, or non-zero tail
    padding.
    """
    expected = packed_element_count(dimensions)
    if expected > MAX_PACKED_ELEMENTS:
        raise analyzer.WorkTooLarge(
            f"packed payload would hold {expected} elements "
            f"(limit {MAX_PACKED_ELEMENTS})"
        )
    # Reject oversized encodings before allocating the decoded buffer. A
    # well-formed base64 string for `expected` bytes is at most 4/3 as long,
    # plus up to two '=' padding characters.
    if len(packed_b64) > (expected + 2) // 3 * 4:
        raise LayoutError(
            f"packed weights base64 holds more than the {expected} expected bytes"
        )

    try:
        raw = base64.b64decode(packed_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LayoutError(f"packed weights are not valid base64: {exc}") from exc

    if len(raw) < expected:
        raise LayoutError(
            f"packed weights hold {len(raw)} bytes, expected {expected} "
            f"(short by {expected - len(raw)})"
        )
    if len(raw) > expected:
        raise LayoutError(
            f"packed weights hold {len(raw)} bytes, expected {expected} "
            f"(extra {len(raw) - expected})"
        )

    extents = axis_extents(dimensions)
    blocks = axis_blocks(dimensions)

    weights: dict[tuple[int, int, int, int], int] = {}
    valid = 0
    for coord, byte in zip(iter_packed_coords(order, extents, blocks), raw):
        value = byte - 256 if byte > 127 else byte  # signed INT8 decode
        if any(coord[axis] >= extents[axis] for axis in AXES):
            if value != 0:
                raise LayoutError(
                    f"packed tail padding must be zero; found {value} at "
                    f"{axis_index(coord)}"
                )
            continue
        valid += 1
        weights[(coord["oc"], coord["cin"], coord["ky"], coord["kx"])] = value
    return weights, valid, len(raw)


# --------------------------------------------------------------------------
# weight comparison and replay
# --------------------------------------------------------------------------

def compare_weights(
    reference: dict[tuple[int, int, int, int], int],
    decoded: dict[tuple[int, int, int, int], int],
    extents: dict[str, int],
) -> dict:
    """Element-wise check in canonical (oc, cin, ky, kx) order."""
    mismatch_count = 0
    first: dict | None = None
    for oc in range(extents["oc"]):
        for cin in range(extents["cin"]):
            for ky in range(extents["ky"]):
                for kx in range(extents["kx"]):
                    key = (oc, cin, ky, kx)
                    want = reference[key]
                    got = decoded[key]
                    if want == got:
                        continue
                    mismatch_count += 1
                    if first is None:
                        first = {
                            "index": {
                                "output_channel": oc,
                                "input_channel": cin,
                                "kernel_y": ky,
                                "kernel_x": kx,
                            },
                            "reference": want,
                            "packed": got,
                        }
    return {"mismatch_count": mismatch_count, "first_mismatch": first}


def replay(
    *,
    feature_map: list[list[list[int]]],
    zero_points: list[int],
    bias: list[int],
    reference: dict[tuple[int, int, int, int], int],
    decoded: dict[tuple[int, int, int, int], int],
    dimensions,
    target_index: tuple[int, int, int, int] | None = None,
) -> dict:
    """Accumulate reference and decoded kernels side by side over the map.

    Semantics are identical to :mod:`analyzer`: stride 1, odd kernel, output
    size == input size, out-of-bounds taps read the channel zero point.

    The whole map is replayed. Two findings are collected:

    * ``first_difference`` -- the first tap (in output scan order) whose two
      running partial sums differ.
    * ``target_occurrence`` -- the first read of ``target_index`` (usually the
      element-wise first mismatch), with its output coordinate and both
      partial sums at that step.
    """
    cout = dimensions.output_channels
    cin = dimensions.input_channels
    kh = dimensions.kernel_height
    kw = dimensions.kernel_width
    height = len(feature_map[0])
    width = len(feature_map[0][0])
    ph, pw = kh // 2, kw // 2
    steps = [
        (c, i, j) for c in range(cin) for i in range(kh) for j in range(kw)
    ]

    work = cout * height * width * len(steps)
    if work > analyzer.MAX_WORK_ITEMS:
        raise analyzer.WorkTooLarge(
            f"replay would take ~{work} accumulate/compare iterations "
            f"(limit {analyzer.MAX_WORK_ITEMS}); reduce channels, kernel, "
            f"feature map or output channels"
        )

    positions = 0
    steps_checked = 0
    first_difference: dict | None = None
    target_occurrence: dict | None = None

    for oc in range(cout):
        for y in range(height):
            for x in range(width):
                positions += 1
                acc_ref = acc_dec = bias[oc]
                for step_no, (c, i, j) in enumerate(steps, start=1):
                    iy = y + i - ph
                    ix = x + j - pw
                    if 0 <= iy < height and 0 <= ix < width:
                        activation = feature_map[c][iy][ix]
                    else:
                        activation = zero_points[c]
                    delta = activation - zero_points[c]
                    key = (oc, c, i, j)
                    w_ref = reference[key]
                    w_dec = decoded[key]
                    term_ref = delta * w_ref
                    term_dec = delta * w_dec
                    acc_ref += term_ref
                    acc_dec += term_dec
                    steps_checked += 1

                    if (
                        target_index is not None
                        and key == target_index
                        and target_occurrence is None
                        and term_ref != term_dec
                    ):
                        target_occurrence = {
                            "output": {"y": y, "x": x, "channel": oc},
                            "step": step_no,
                            "activation": activation,
                            "zero_point": zero_points[c],
                            "delta": delta,
                            "reference_weight": w_ref,
                            "packed_weight": w_dec,
                            "reference_contribution": term_ref,
                            "packed_contribution": term_dec,
                            "reference_partial_sum": acc_ref,
                            "packed_partial_sum": acc_dec,
                        }

                    if first_difference is None and acc_ref != acc_dec:
                        first_difference = {
                            "index": {
                                "output_channel": oc,
                                "input_channel": c,
                                "kernel_y": i,
                                "kernel_x": j,
                            },
                            "output": {"y": y, "x": x, "channel": oc},
                            "step": step_no,
                            "reference_weight": w_ref,
                            "packed_weight": w_dec,
                            "activation": activation,
                            "zero_point": zero_points[c],
                            "delta": delta,
                            "reference_contribution": term_ref,
                            "packed_contribution": term_dec,
                            "reference_partial_sum": acc_ref,
                            "packed_partial_sum": acc_dec,
                        }

    return {
        "positions_checked": positions,
        "steps_checked": steps_checked,
        "first_difference": first_difference,
        "target_occurrence": target_occurrence,
    }


# --------------------------------------------------------------------------
# top-level check
# --------------------------------------------------------------------------

def check_layout(
    *,
    feature_map_data: list[list[list[int]]],
    feature_map_height: int,
    feature_map_width: int,
    zero_points: list[int],
    reference_weights: list,
    bias: list[int],
    dimensions,
    decoded_axis_order: list[str],
    packed_axis_order: list[str],
    packed_base64: str,
) -> dict:
    """Decode the device buffer, compare weights, replay, and report."""
    decoded_axis_order = check_axis_order(decoded_axis_order, "decoded_axis_order")
    packed_axis_order = check_axis_order(packed_axis_order, "packed_axis_order")

    extents = axis_extents(dimensions)
    if len(feature_map_data) != extents["cin"]:
        raise LayoutError(
            f"feature map has {len(feature_map_data)} channels, expected "
            f"{extents['cin']}"
        )
    if len(feature_map_data[0]) != feature_map_height or len(feature_map_data[0][0]) != feature_map_width:
        raise LayoutError(
            f"feature map channel shape is "
            f"{len(feature_map_data[0])}x{len(feature_map_data[0][0])}, expected "
            f"{feature_map_height}x{feature_map_width}"
        )

    reference = decode_nested(reference_weights, decoded_axis_order, extents)
    decoded, valid_elements, packed_bytes = unpack_packed(
        packed_base64, packed_axis_order, dimensions
    )

    weight_check = compare_weights(reference, decoded, extents)
    first_weight = weight_check["first_mismatch"]
    target_index = None
    if first_weight is not None:
        idx = first_weight["index"]
        target_index = (
            idx["output_channel"], idx["input_channel"],
            idx["kernel_y"], idx["kernel_x"],
        )
    replay_result = replay(
        feature_map=feature_map_data,
        zero_points=zero_points,
        bias=bias,
        reference=reference,
        decoded=decoded,
        dimensions=dimensions,
        target_index=target_index,
    )

    first_replay = replay_result["first_difference"]
    target_occurrence = replay_result["target_occurrence"]

    # Order of reporting: the element-wise weight check runs first, so its first
    # mismatch leads. The replay then supplies the output coordinate that first
    # reads that tap and both partial sums at that step. If the given feature map
    # never reads the bad tap (for example, its delta is 0), those stay null.
    # A replay divergence implies a weight mismatch -- identical kernels cannot
    # diverge -- so `first_replay` without `first_weight` is unreachable and is
    # asserted rather than reported.
    difference = None
    if first_weight is not None:
        difference = {
            "kind": "weight_mismatch",
            "index": first_weight["index"],
            "reference_weight": first_weight["reference"],
            "packed_weight": first_weight["packed"],
            "output": target_occurrence["output"] if target_occurrence else None,
            "step": target_occurrence["step"] if target_occurrence else None,
            "reference_partial_sum": (
                target_occurrence["reference_partial_sum"] if target_occurrence else None
            ),
            "packed_partial_sum": (
                target_occurrence["packed_partial_sum"] if target_occurrence else None
            ),
        }
    else:
        assert first_replay is None, "replay diverged with identical kernels"
    weights_match = first_weight is None
    replay_match = first_replay is None
    # Divergence implies a weight mismatch (equal kernels accumulate equally);
    # the converse is allowed, the map may never read the bad tap.
    assert replay_match or not weights_match, "replay cannot diverge on equal weights"
    return {
        "matches": weights_match and replay_match,
        "weights_match": weights_match,
        "replay_match": replay_match,
        "layout": {
            "decoded_axis_order": decoded_axis_order,
            "packed_axis_order": packed_axis_order,
            "output_channel_block": dimensions.output_channel_block,
            "input_channel_block": dimensions.input_channel_block,
            "kernel_block": dimensions.kernel_block,
            "signed_int8": True,
        },
        "packed": {
            "bytes": packed_bytes,
            "valid_elements": valid_elements,
            "padding_elements": packed_element_count(dimensions) - valid_elements,
        },
        "weight_check": weight_check,
        "replay": replay_result,
        "difference": difference,
    }
