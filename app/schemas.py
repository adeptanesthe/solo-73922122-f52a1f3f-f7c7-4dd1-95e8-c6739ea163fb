"""Request schema for the single-layer accumulator inspection API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

INT8_MIN, INT8_MAX = -128, 127
UINT8_MIN, UINT8_MAX = 0, 255
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
MAX_KERNEL = 31


def validate_weight_tensor(
    weights: list[list[list[list[int]]]],
    *,
    expected_input_channels: int | None = None,
    input_channel_source: str = "",
) -> tuple[int, int, int]:
    """Validate a nested ``[out_ch][in_ch][ky][kx]`` INT8 kernel tensor.

    Returns ``(input_channels, kernel_height, kernel_width)``. The tensor must
    be rectangular, in the signed INT8 range, and odd in both kernel dims.
    """
    cout = len(weights)
    if cout == 0:
        raise ValueError("weights must contain at least one output channel")

    kh = kw = None
    cin = None
    for oc, w3 in enumerate(weights):
        if expected_input_channels is not None and len(w3) != expected_input_channels:
            suffix = f" ({input_channel_source})" if input_channel_source else ""
            raise ValueError(
                f"weights[{oc}] has {len(w3)} input channels, expected "
                f"{expected_input_channels}{suffix}"
            )
        if len(w3) == 0:
            raise ValueError(f"weights[{oc}] has no input channels")
        if cin is None:
            cin = len(w3)
        elif len(w3) != cin:
            raise ValueError(
                f"weights[{oc}] has {len(w3)} input channels, expected {cin}"
            )

        for c, w2 in enumerate(w3):
            if kh is None:
                kh = len(w2)
                if kh == 0:
                    raise ValueError("kernel height must be >= 1")
                kw = len(w2[0]) if w2 else 0
                if kw == 0:
                    raise ValueError("kernel width must be >= 1")
            if len(w2) != kh:
                raise ValueError(f"weights[{oc}][{c}] has {len(w2)} rows, expected {kh}")
            for i, row in enumerate(w2):
                if len(row) != kw:
                    raise ValueError(f"weights[{oc}][{c}][{i}] has {len(row)} cols, expected {kw}")
                for j, v in enumerate(row):
                    if not INT8_MIN <= v <= INT8_MAX:
                        raise ValueError(
                            f"weights[{oc}][{c}][{i}][{j}]={v} outside INT8 range [-128, 127]"
                        )

    if kh is None or kw is None:
        raise ValueError("weights must contain at least one input channel")
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"kernel must be odd in both dims, got {kh}x{kw}")
    if kh > MAX_KERNEL or kw > MAX_KERNEL:
        raise ValueError(f"kernel {kh}x{kw} exceeds limit {MAX_KERNEL}")
    return cin, kh, kw


class FeatureMapSpec(BaseModel):
    height: int = Field(ge=1, le=1024, description="input feature map height")
    width: int = Field(ge=1, le=1024, description="input feature map width")


class ChannelSpec(BaseModel):
    min: int = Field(ge=UINT8_MIN, le=UINT8_MAX, description="min uint8 activation of this channel")
    max: int = Field(ge=UINT8_MIN, le=UINT8_MAX, description="max uint8 activation of this channel")
    zero_point: int = Field(ge=UINT8_MIN, le=UINT8_MAX, description="uint8 zero point of this channel")

    @model_validator(mode="after")
    def _range_ok(self) -> "ChannelSpec":
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class AnalyzeRequest(BaseModel):
    """One INT8 conv layer, stride 1, odd kernel, zero-point padding, same output size."""

    feature_map: FeatureMapSpec
    activations: list[ChannelSpec] = Field(
        min_length=1, max_length=1024, description="per-input-channel uint8 range + zero point"
    )
    weights: list[list[list[list[int]]]] = Field(
        min_length=1, max_length=1024,
        description="INT8 kernel [out_channel][in_channel][ky][kx]",
    )
    bias: list[int] = Field(min_length=1, description="per-output-channel integer bias (accumulator step 0)")
    target_bits: int = Field(ge=1, le=64, description="signed accumulator width under evaluation")

    @model_validator(mode="after")
    def _check_shapes_and_ranges(self) -> "AnalyzeRequest":
        cout = len(self.weights)
        if len(self.bias) != cout:
            raise ValueError(f"bias has {len(self.bias)} entries but weights have {cout} output channels")
        for oc, b in enumerate(self.bias):
            if not INT64_MIN <= b <= INT64_MAX:
                raise ValueError(f"bias[{oc}]={b} outside int64 range")


        validate_weight_tensor(
            self.weights,
            expected_input_channels=len(self.activations),
            input_channel_source="= len(activations)",
        )
        return self


# --------------------------------------------------------------------------
# weight-layout equivalence check
# --------------------------------------------------------------------------

class LayoutDimensions(BaseModel):
    """Extent and blocking of every packed axis (device-side layout)."""

    output_channels: int = Field(ge=1, le=1024, description="number of output channels")
    input_channels: int = Field(ge=1, le=1024, description="number of input channels")
    kernel_height: int = Field(ge=1, le=MAX_KERNEL, description="odd kernel height")
    kernel_width: int = Field(ge=1, le=MAX_KERNEL, description="odd kernel width")
    output_channel_block: int = Field(ge=1, le=1024, description="oc axis block size")
    input_channel_block: int = Field(ge=1, le=1024, description="cin axis block size")
    kernel_block: int = Field(ge=1, le=MAX_KERNEL, description="ky and kx axis block size")

    @model_validator(mode="after")
    def _odd_kernel(self) -> "LayoutDimensions":
        if self.kernel_height % 2 == 0 or self.kernel_width % 2 == 0:
            raise ValueError(
                f"kernel must be odd in both dims, got "
                f"{self.kernel_height}x{self.kernel_width}"
            )
        return self


class WeightLayoutSpec(BaseModel):
    decoded_axis_order: list[str] = Field(
        min_length=4, max_length=4,
        description="axis order of the reference weights, a permutation of oc/cin/ky/kx",
    )
    packed_axis_order: list[str] = Field(
        min_length=4, max_length=4,
        description="axis order of the packed device buffer, a permutation of oc/cin/ky/kx",
    )

    @model_validator(mode="after")
    def _permutations(self) -> "WeightLayoutSpec":
        for field, order in (
            ("decoded_axis_order", self.decoded_axis_order),
            ("packed_axis_order", self.packed_axis_order),
        ):
            if len(order) != 4 or set(order) != {"oc", "cin", "ky", "kx"}:
                raise ValueError(f"{field} must be a permutation of oc/cin/ky/kx, got {order}")
        return self


class FeatureMapData(BaseModel):
    """One concrete uint8 feature map: ``[in_ch][height][width]``."""

    height: int = Field(ge=1, le=1024, description="feature map height")
    width: int = Field(ge=1, le=1024, description="feature map width")
    data: list[list[list[int]]] = Field(
        min_length=1, max_length=1024,
        description="uint8 values, nested [in_channel][y][x]",
    )

    @model_validator(mode="after")
    def _shape_and_range(self) -> "FeatureMapData":
        for c, plane in enumerate(self.data):
            if len(plane) != self.height:
                raise ValueError(
                    f"feature_map.data[{c}] has {len(plane)} rows, expected {self.height}"
                )
            for y, row in enumerate(plane):
                if len(row) != self.width:
                    raise ValueError(
                        f"feature_map.data[{c}][{y}] has {len(row)} cols, expected {self.width}"
                    )
                for x, v in enumerate(row):
                    if not UINT8_MIN <= v <= UINT8_MAX:
                        raise ValueError(
                            f"feature_map.data[{c}][{y}][{x}]={v} outside uint8 range [0, 255]"
                        )
        cells = len(self.data) * self.height * self.width
        if cells > 4_000_000:
            raise ValueError(
                f"feature map has {cells} cells (limit 4000000)"
            )
        return self


class LayoutCheckRequest(BaseModel):
    """Compare a device packed kernel buffer against reference weights."""

    weights: list[list[list[list[int]]]] = Field(
        min_length=1, max_length=1024,
        description="reference INT8 kernel, decoded_axis_order layout",
    )
    bias: list[int] = Field(min_length=1, description="per-output-channel integer bias")
    feature_map: FeatureMapData
    input_zero_points: list[int] = Field(
        min_length=1, max_length=1024,
        description="per-input-channel uint8 zero point",
    )
    dimensions: LayoutDimensions
    layout: WeightLayoutSpec
    packed_weights_base64: str = Field(
        min_length=1, description="device packed INT8 kernel, base64-encoded"
    )

    @model_validator(mode="after")
    def _check(self) -> "LayoutCheckRequest":
        cin = self.dimensions.input_channels
        cout = self.dimensions.output_channels
        if len(self.input_zero_points) != cin:
            raise ValueError(
                f"input_zero_points has {len(self.input_zero_points)} entries, "
                f"expected {cin} (= dimensions.input_channels)"
            )
        for c, zp in enumerate(self.input_zero_points):
            if not UINT8_MIN <= zp <= UINT8_MAX:
                raise ValueError(f"input_zero_points[{c}]={zp} outside uint8 range [0, 255]")
        if len(self.feature_map.data) != cin:
            raise ValueError(
                f"feature_map has {len(self.feature_map.data)} channels, expected {cin}"
            )
        if len(self.bias) != cout:
            raise ValueError(
                f"bias has {len(self.bias)} entries but dimensions.output_channels is {cout}"
            )
        for oc, b in enumerate(self.bias):
            if not INT64_MIN <= b <= INT64_MAX:
                raise ValueError(f"bias[{oc}]={b} outside int64 range")

        # Reference weights are stored in decoded_axis_order; check extents
        # along that order and the signed INT8 value range at every leaf.
        extents = {
            "oc": cout,
            "cin": cin,
            "ky": self.dimensions.kernel_height,
            "kx": self.dimensions.kernel_width,
        }
        _check_nested_extent(self.weights, self.layout.decoded_axis_order, 0, extents)
        return self


def _check_nested_extent(node, order: list[str], depth: int, extents: dict[str, int]) -> None:
    axis = order[depth]
    if not isinstance(node, list):
        raise ValueError(f"weights axis '{axis}' must be a list")
    want = extents[axis]
    if len(node) != want:
        raise ValueError(
            f"weights axis '{axis}' has length {len(node)}, expected {want}"
        )
    if depth < 3:
        for child in node:
            _check_nested_extent(child, order, depth + 1, extents)
        return
    for i, v in enumerate(node):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(
                f"weights axis '{axis}' index {i} is {v!r}, not an integer"
            )
        if not INT8_MIN <= v <= INT8_MAX:
            raise ValueError(
                f"weights axis '{axis}' index {i} = {v} outside INT8 range [-128, 127]"
            )
