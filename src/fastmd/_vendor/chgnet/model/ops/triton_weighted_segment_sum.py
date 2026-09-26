"""Fused edge weighting and segment reduction for CHGNet convolutions."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _weighted_segsum_fwd(
    weight,
    value,
    segment,
    out,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    SEGMENT_STRIDE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_d * BD + tl.arange(0, BD)
    row_mask = rows < N
    mask = row_mask[:, None] & (cols[None, :] < D)
    owner = tl.load(segment + rows * SEGMENT_STRIDE, mask=row_mask, other=0)
    offsets = rows[:, None] * D + cols[None, :]
    weights = tl.load(weight + offsets, mask=mask, other=0.0)
    values = tl.load(value + offsets, mask=mask, other=0.0)
    tl.atomic_add(
        out + owner[:, None] * D + cols[None, :],
        weights * values,
        mask=mask,
        sem="relaxed",
    )


@triton.jit
def _weighted_segsum_bwd(
    grad_out,
    weight,
    value,
    segment,
    grad_weight,
    grad_value,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    SEGMENT_STRIDE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_d * BD + tl.arange(0, BD)
    row_mask = rows < N
    mask = row_mask[:, None] & (cols[None, :] < D)
    owner = tl.load(segment + rows * SEGMENT_STRIDE, mask=row_mask, other=0)
    offsets = rows[:, None] * D + cols[None, :]
    grad = tl.load(
        grad_out + owner[:, None] * D + cols[None, :],
        mask=mask,
        other=0.0,
    )
    weights = tl.load(weight + offsets, mask=mask, other=0.0)
    values = tl.load(value + offsets, mask=mask, other=0.0)
    tl.store(grad_weight + offsets, grad * values, mask=mask)
    tl.store(grad_value + offsets, grad * weights, mask=mask)


class _WeightedSegmentSum(Function):
    @staticmethod
    def forward(ctx, weight, value, segment, num_segment):
        weight = weight.contiguous()
        value = value.contiguous()
        rows, dim = weight.shape
        block_rows = 4
        block_dim = min(triton.next_power_of_2(dim), 128)
        out = value.new_zeros((num_segment, dim))
        grid = (
            triton.cdiv(rows, block_rows),
            triton.cdiv(dim, block_dim),
        )
        _weighted_segsum_fwd[grid](
            weight,
            value,
            segment,
            out,
            N=rows,
            D=dim,
            BM=block_rows,
            BD=block_dim,
            SEGMENT_STRIDE=segment.stride(0),
            num_warps=4,
        )
        ctx.save_for_backward(weight, value, segment)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight, value, segment = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        rows, dim = weight.shape
        block_rows = 4
        block_dim = min(triton.next_power_of_2(dim), 128)
        grad_weight = torch.empty_like(weight)
        grad_value = torch.empty_like(value)
        grid = (
            triton.cdiv(rows, block_rows),
            triton.cdiv(dim, block_dim),
        )
        _weighted_segsum_bwd[grid](
            grad_out,
            weight,
            value,
            segment,
            grad_weight,
            grad_value,
            N=rows,
            D=dim,
            BM=block_rows,
            BD=block_dim,
            SEGMENT_STRIDE=segment.stride(0),
            num_warps=4,
        )
        return grad_weight, grad_value, None, None


def weighted_segment_sum(
    weight: torch.Tensor,
    value: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
) -> torch.Tensor:
    """Compute ``out[segment] += weight * value`` without materializing product."""
    if weight.ndim != 2 or value.ndim != 2:
        raise ValueError("weight and value must be rank-2 matrices")
    if weight.shape != value.shape:
        raise ValueError("weight and value must have the same shape")
    if not weight.is_floating_point() or not value.is_floating_point():
        raise TypeError("weight and value must use floating-point dtypes")
    if weight.dtype != value.dtype:
        raise TypeError("weight and value must have the same dtype")
    if segment.ndim != 1 or segment.shape[0] != weight.shape[0]:
        raise ValueError("segment must have one entry per input row")
    if segment.dtype not in (torch.int32, torch.int64):
        raise TypeError("segment must use int32 or int64 indices")
    if num_segment < 0:
        raise ValueError("num_segment must be non-negative")
    if num_segment == 0 and weight.shape[0] > 0:
        raise ValueError("num_segment must be positive when input rows are present")
    if not (weight.device == value.device == segment.device):
        raise ValueError("weight, value, and segment must be on the same device")
    if weight.is_cuda and weight.shape[0] > 0 and weight.shape[1] > 0:
        return _WeightedSegmentSum.apply(weight, value, segment, int(num_segment))
    product = weight * value
    return product.new_zeros((int(num_segment), value.shape[1])).index_add_(
        0, segment, product
    )


@triton.jit
def _triple_weighted_segsum_fwd(
    weight_i,
    weight_j,
    value,
    segment,
    out,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    SEGMENT_STRIDE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_d * BD + tl.arange(0, BD)
    row_mask = rows < N
    mask = row_mask[:, None] & (cols[None, :] < D)
    owner = tl.load(segment + rows * SEGMENT_STRIDE, mask=row_mask, other=0)
    offsets = rows[:, None] * D + cols[None, :]
    left = tl.load(weight_i + offsets, mask=mask, other=0.0)
    right = tl.load(weight_j + offsets, mask=mask, other=0.0)
    values = tl.load(value + offsets, mask=mask, other=0.0)
    tl.atomic_add(
        out + owner[:, None] * D + cols[None, :],
        left * right * values,
        mask=mask,
        sem="relaxed",
    )


@triton.jit
def _triple_weighted_segsum_bwd(
    grad_out,
    weight_i,
    weight_j,
    value,
    segment,
    grad_weight_i,
    grad_weight_j,
    grad_value,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    SEGMENT_STRIDE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_d * BD + tl.arange(0, BD)
    row_mask = rows < N
    mask = row_mask[:, None] & (cols[None, :] < D)
    owner = tl.load(segment + rows * SEGMENT_STRIDE, mask=row_mask, other=0)
    offsets = rows[:, None] * D + cols[None, :]
    grad = tl.load(
        grad_out + owner[:, None] * D + cols[None, :],
        mask=mask,
        other=0.0,
    )
    left = tl.load(weight_i + offsets, mask=mask, other=0.0)
    right = tl.load(weight_j + offsets, mask=mask, other=0.0)
    values = tl.load(value + offsets, mask=mask, other=0.0)
    tl.store(grad_weight_i + offsets, grad * right * values, mask=mask)
    tl.store(grad_weight_j + offsets, grad * left * values, mask=mask)
    tl.store(grad_value + offsets, grad * left * right, mask=mask)


class _TripleWeightedSegmentSum(Function):
    @staticmethod
    def forward(ctx, weight_i, weight_j, value, segment, num_segment):
        weight_i = weight_i.contiguous()
        weight_j = weight_j.contiguous()
        value = value.contiguous()
        rows, dim = value.shape
        block_rows = 4
        block_dim = min(triton.next_power_of_2(dim), 128)
        out = value.new_zeros((num_segment, dim))
        grid = (
            triton.cdiv(rows, block_rows),
            triton.cdiv(dim, block_dim),
        )
        _triple_weighted_segsum_fwd[grid](
            weight_i,
            weight_j,
            value,
            segment,
            out,
            N=rows,
            D=dim,
            BM=block_rows,
            BD=block_dim,
            SEGMENT_STRIDE=segment.stride(0),
            num_warps=4,
        )
        ctx.save_for_backward(weight_i, weight_j, value, segment)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight_i, weight_j, value, segment = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        rows, dim = value.shape
        block_rows = 4
        block_dim = min(triton.next_power_of_2(dim), 128)
        grad_weight_i = torch.empty_like(weight_i)
        grad_weight_j = torch.empty_like(weight_j)
        grad_value = torch.empty_like(value)
        grid = (
            triton.cdiv(rows, block_rows),
            triton.cdiv(dim, block_dim),
        )
        _triple_weighted_segsum_bwd[grid](
            grad_out,
            weight_i,
            weight_j,
            value,
            segment,
            grad_weight_i,
            grad_weight_j,
            grad_value,
            N=rows,
            D=dim,
            BM=block_rows,
            BD=block_dim,
            SEGMENT_STRIDE=segment.stride(0),
            num_warps=4,
        )
        return grad_weight_i, grad_weight_j, grad_value, None, None


def triple_weighted_segment_sum(
    weight_i: torch.Tensor,
    weight_j: torch.Tensor,
    value: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
) -> torch.Tensor:
    """Compute ``out[segment] += weight_i * weight_j * value`` in one kernel."""
    if weight_i.ndim != 2 or not (
        weight_i.shape == weight_j.shape == value.shape
    ):
        raise ValueError("weight_i, weight_j, and value must be equal matrices")
    if not weight_i.is_floating_point() or not (
        weight_i.dtype == weight_j.dtype == value.dtype
    ):
        raise TypeError("weight_i, weight_j, and value must share a floating dtype")
    if segment.ndim != 1 or segment.shape[0] != value.shape[0]:
        raise ValueError("segment must have one entry per input row")
    if segment.dtype not in (torch.int32, torch.int64):
        raise TypeError("segment must use int32 or int64 indices")
    if num_segment < 0 or (num_segment == 0 and value.shape[0] > 0):
        raise ValueError("num_segment is incompatible with the input rows")
    if any(
        tensor.device != value.device
        for tensor in (weight_i, weight_j, segment)
    ):
        raise ValueError("all triple segment-sum inputs must share one device")
    if value.is_cuda and value.shape[0] > 0 and value.shape[1] > 0:
        return _TripleWeightedSegmentSum.apply(
            weight_i,
            weight_j,
            value,
            segment,
            int(num_segment),
        )
    product = weight_i * weight_j * value
    return product.new_zeros((int(num_segment), value.shape[1])).index_add_(
        0, segment, product
    )
