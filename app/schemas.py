"""Request schema for the single-layer accumulator inspection API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

INT8_MIN, INT8_MAX = -128, 127
UINT8_MIN, UINT8_MAX = 0, 255
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
MAX_KERNEL = 31


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
        cin = len(self.activations)
        cout = len(self.weights)
        if len(self.bias) != cout:
            raise ValueError(f"bias has {len(self.bias)} entries but weights have {cout} output channels")
        for oc, b in enumerate(self.bias):
            if not INT64_MIN <= b <= INT64_MAX:
                raise ValueError(f"bias[{oc}]={b} outside int64 range")

        kh = kw = None
        for oc, w3 in enumerate(self.weights):
            if len(w3) != cin:
                raise ValueError(
                    f"weights[{oc}] has {len(w3)} input channels, expected {cin} (= len(activations))"
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
        assert kh is not None and kw is not None
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(f"kernel must be odd in both dims, got {kh}x{kw}")
        if kh > MAX_KERNEL or kw > MAX_KERNEL:
            raise ValueError(f"kernel {kh}x{kw} exceeds limit {MAX_KERNEL}")
        return self
