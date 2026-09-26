"""Fused indexed weighting and segment reduction for CHGNet bond updates."""

from __future__ import annotations

from numbers import Integral

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _indexed_triple_weighted_segsum_fwd(
    weight,
    value,
    left,
    right,
    segment,
    out,
    N_ROWS: tl.constexpr,
    N_FEATURES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    WEIGHT_ROW_STRIDE: tl.constexpr,
    WEIGHT_FEATURE_STRIDE: tl.constexpr,
    VALUE_ROW_STRIDE: tl.constexpr,
    VALUE_FEATURE_STRIDE: tl.constexpr,
    LEFT_STRIDE: tl.constexpr,
    RIGHT_STRIDE: tl.constexpr,
    SEGMENT_STRIDE: tl.constexpr,
):
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    features = (
        feature_block * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    )
    row_mask = rows < N_ROWS
    mask = row_mask[:, None] & (features[None, :] < N_FEATURES)

    left_index = tl.load(left + rows * LEFT_STRIDE, mask=row_mask, other=0)
    right_index = tl.load(right + rows * RIGHT_STRIDE, mask=row_mask, other=0)
    owner = tl.load(segment + rows * SEGMENT_STRIDE, mask=row_mask, other=0)

    left_weight = tl.load(
        weight
        + left_index[:, None] * WEIGHT_ROW_STRIDE
        + features[None, :] * WEIGHT_FEATURE_STRIDE,
        mask=mask,
        other=0.0,
    )
    right_weight = tl.load(
        weight
        + right_index[:, None] * WEIGHT_ROW_STRIDE
        + features[None, :] * WEIGHT_FEATURE_STRIDE,
        mask=mask,
        other=0.0,
    )
    values = tl.load(
        value
        + rows[:, None] * VALUE_ROW_STRIDE
        + features[None, :] * VALUE_FEATURE_STRIDE,
        mask=mask,
        other=0.0,
    )
    tl.atomic_add(
        out + owner[:, None] * N_FEATURES + features[None, :],
        left_weight * right_weight * values,
        mask=mask,
        sem="relaxed",
    )


@triton.jit
def _indexed_triple_weighted_segsum_bwd(
    grad_out,
    weight,
    value,
    left,
    right,
    segment,
    grad_weight,
    grad_value,
    N_ROWS: tl.constexpr,
    N_FEATURES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    WEIGHT_ROW_STRIDE: tl.constexpr,
    WEIGHT_FEATURE_STRIDE: tl.constexpr,
    VALUE_ROW_STRIDE: tl.constexpr,
    VALUE_FEATURE_STRIDE: tl.constexpr,
    LEFT_STRIDE: tl.constexpr,
    RIGHT_STRIDE: tl.constexpr,
    SEGMENT_STRIDE: tl.constexpr,
    NEED_WEIGHT_GRAD: tl.constexpr,
    NEED_VALUE_GRAD: tl.constexpr,
):
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    features = (
        feature_block * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    )
    row_mask = rows < N_ROWS
    mask = row_mask[:, None] & (features[None, :] < N_FEATURES)

    left_index = tl.load(left + rows * LEFT_STRIDE, mask=row_mask, other=0)
    right_index = tl.load(right + rows * RIGHT_STRIDE, mask=row_mask, other=0)
    owner = tl.load(segment + rows * SEGMENT_STRIDE, mask=row_mask, other=0)
    left_offsets = (
        left_index[:, None] * WEIGHT_ROW_STRIDE
        + features[None, :] * WEIGHT_FEATURE_STRIDE
    )
    right_offsets = (
        right_index[:, None] * WEIGHT_ROW_STRIDE
        + features[None, :] * WEIGHT_FEATURE_STRIDE
    )
    left_weight = tl.load(weight + left_offsets, mask=mask, other=0.0)
    right_weight = tl.load(weight + right_offsets, mask=mask, other=0.0)
    values = tl.load(
        value
        + rows[:, None] * VALUE_ROW_STRIDE
        + features[None, :] * VALUE_FEATURE_STRIDE,
        mask=mask,
        other=0.0,
    )
    grad = tl.load(
        grad_out + owner[:, None] * N_FEATURES + features[None, :],
        mask=mask,
        other=0.0,
    )

    if NEED_WEIGHT_GRAD:
        # Keep these as two atomic operations. When left_index == right_index,
        # both product-rule terms must be accumulated into the same row.
        tl.atomic_add(
            grad_weight + left_index[:, None] * N_FEATURES + features[None, :],
            grad * right_weight * values,
            mask=mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_weight + right_index[:, None] * N_FEATURES + features[None, :],
            grad * left_weight * values,
            mask=mask,
            sem="relaxed",
        )
    if NEED_VALUE_GRAD:
        tl.store(
            grad_value + rows[:, None] * N_FEATURES + features[None, :],
            grad * left_weight * right_weight,
            mask=mask,
        )


def _native_indexed_triple_weighted_segment_sum(
    weight: torch.Tensor,
    value: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
) -> torch.Tensor:
    product = (
        weight.index_select(0, left)
        * weight.index_select(0, right)
        * value
    )
    return product.new_zeros((num_segment, value.shape[1])).index_add(
        0,
        segment,
        product,
    )


def _native_backward(
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    value: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    segment: torch.Tensor,
    *,
    need_weight_grad: bool,
    need_value_grad: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    grad = grad_out.index_select(0, segment)
    left_weight = weight.index_select(0, left)
    right_weight = weight.index_select(0, right)

    grad_weight = None
    if need_weight_grad:
        left_contribution = grad * right_weight * value
        right_contribution = grad * left_weight * value
        grad_weight = torch.zeros_like(
            weight,
            memory_format=torch.contiguous_format,
        )
        grad_weight = grad_weight.index_add(0, left, left_contribution)
        grad_weight = grad_weight.index_add(0, right, right_contribution)

    grad_value = None
    if need_value_grad:
        grad_value = grad * left_weight * right_weight
    return grad_weight, grad_value


class _IndexedTripleWeightedSegmentSum(Function):
    @staticmethod
    def forward(ctx, weight, value, left, right, segment, num_segment):
        rows, width = value.shape
        block_rows = 4
        block_features = min(triton.next_power_of_2(width), 128)
        out = value.new_zeros((num_segment, width))
        grid = (
            triton.cdiv(rows, block_rows),
            triton.cdiv(width, block_features),
        )
        _indexed_triple_weighted_segsum_fwd[grid](
            weight,
            value,
            left,
            right,
            segment,
            out,
            N_ROWS=rows,
            N_FEATURES=width,
            BLOCK_ROWS=block_rows,
            BLOCK_FEATURES=block_features,
            WEIGHT_ROW_STRIDE=weight.stride(0),
            WEIGHT_FEATURE_STRIDE=weight.stride(1),
            VALUE_ROW_STRIDE=value.stride(0),
            VALUE_FEATURE_STRIDE=value.stride(1),
            LEFT_STRIDE=left.stride(0),
            RIGHT_STRIDE=right.stride(0),
            SEGMENT_STRIDE=segment.stride(0),
            num_warps=4,
        )
        ctx.save_for_backward(weight, value, left, right, segment)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight, value, left, right, segment = ctx.saved_tensors
        need_weight_grad = ctx.needs_input_grad[0]
        need_value_grad = ctx.needs_input_grad[1]

        # create_graph=True enables grad mode inside backward. Use PyTorch in
        # that case so higher-order derivatives retain a differentiable graph.
        if torch.is_grad_enabled():
            grad_weight, grad_value = _native_backward(
                grad_out,
                weight,
                value,
                left,
                right,
                segment,
                need_weight_grad=need_weight_grad,
                need_value_grad=need_value_grad,
            )
            return grad_weight, grad_value, None, None, None, None

        grad_out = grad_out.contiguous()
        rows, width = value.shape
        block_rows = 4
        block_features = min(triton.next_power_of_2(width), 128)
        grad_weight = (
            torch.zeros_like(weight, memory_format=torch.contiguous_format)
            if need_weight_grad
            else None
        )
        grad_value = (
            torch.empty_like(value, memory_format=torch.contiguous_format)
            if need_value_grad
            else None
        )
        # Triton pointer arguments cannot be Python ``None`` on every supported
        # release. The constexpr guards make these harmless aliases when the
        # corresponding gradient is not requested.
        grad_weight_buffer = grad_weight if grad_weight is not None else grad_out
        grad_value_buffer = grad_value if grad_value is not None else grad_out
        grid = (
            triton.cdiv(rows, block_rows),
            triton.cdiv(width, block_features),
        )
        _indexed_triple_weighted_segsum_bwd[grid](
            grad_out,
            weight,
            value,
            left,
            right,
            segment,
            grad_weight_buffer,
            grad_value_buffer,
            N_ROWS=rows,
            N_FEATURES=width,
            BLOCK_ROWS=block_rows,
            BLOCK_FEATURES=block_features,
            WEIGHT_ROW_STRIDE=weight.stride(0),
            WEIGHT_FEATURE_STRIDE=weight.stride(1),
            VALUE_ROW_STRIDE=value.stride(0),
            VALUE_FEATURE_STRIDE=value.stride(1),
            LEFT_STRIDE=left.stride(0),
            RIGHT_STRIDE=right.stride(0),
            SEGMENT_STRIDE=segment.stride(0),
            NEED_WEIGHT_GRAD=need_weight_grad,
            NEED_VALUE_GRAD=need_value_grad,
            num_warps=4,
        )
        return grad_weight, grad_value, None, None, None, None


def _validate_index_bounds(
    index: torch.Tensor,
    upper_bound: int,
    name: str,
) -> None:
    """Synchronously validate bounds only for CPU tensors."""
    if index.is_cuda or index.numel() == 0:
        return
    minimum = int(index.min())
    maximum = int(index.max())
    if minimum < 0 or maximum >= upper_bound:
        raise IndexError(
            f"{name} entries must be in [0, {upper_bound}); "
            f"found range [{minimum}, {maximum}]"
        )


def indexed_triple_weighted_segment_sum(
    weight: torch.Tensor,
    value: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
) -> torch.Tensor:
    """Fuse indexed bond weighting with a triplet-to-atom segment reduction.

    Computes ``out[segment[t]] += weight[left[t]] * weight[right[t]] * value[t]``.
    CUDA FP32 inputs use Triton for the forward pass and ordinary first-order
    backward. CPU, non-FP32, empty, and higher-order paths use native PyTorch.
    Index tensors may be non-contiguous one-dimensional column views.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be a rank-2 matrix")
    if value.ndim != 2:
        raise ValueError("value must be a rank-2 matrix")
    if weight.shape[1] != value.shape[1]:
        raise ValueError("weight and value must have the same feature width")
    if not weight.is_floating_point() or not value.is_floating_point():
        raise TypeError("weight and value must use floating-point dtypes")
    if weight.dtype != value.dtype:
        raise TypeError("weight and value must have the same dtype")

    rows = value.shape[0]
    for name, index in (
        ("left", left),
        ("right", right),
        ("segment", segment),
    ):
        if index.ndim != 1 or index.shape[0] != rows:
            raise ValueError(f"{name} must be rank 1 with one entry per value row")
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must use int32 or int64 indices")

    if any(
        tensor.device != value.device
        for tensor in (weight, left, right, segment)
    ):
        raise ValueError("weight, value, and all indices must share one device")
    if isinstance(num_segment, bool) or not isinstance(num_segment, Integral):
        raise TypeError("num_segment must be an integer")
    num_segment = int(num_segment)
    if num_segment < 0:
        raise ValueError("num_segment must be non-negative")
    if rows > 0 and num_segment == 0:
        raise ValueError("num_segment must be positive when value has rows")
    if rows > 0 and weight.shape[0] == 0:
        raise ValueError("weight must have rows when value has rows")

    _validate_index_bounds(left, weight.shape[0], "left")
    _validate_index_bounds(right, weight.shape[0], "right")
    _validate_index_bounds(segment, num_segment, "segment")

    use_triton = (
        value.is_cuda
        and value.dtype == torch.float32
        and rows > 0
        and value.shape[1] > 0
    )
    if use_triton:
        return _IndexedTripleWeightedSegmentSum.apply(
            weight,
            value,
            left,
            right,
            segment,
            num_segment,
        )
    return _native_indexed_triple_weighted_segment_sum(
        weight,
        value,
        left,
        right,
        segment,
        num_segment,
    )
