"""Fused reductions for ALIGNN's edge-gated graph convolution.

The MatRIS weighted-segment-sum fusion computes one weighted reduction.
ALIGNN needs the weighted numerator and gate denominator for the same segment,
so this specialization produces both in one launch and avoids materializing
``transformed[source]``, sigmoid gates, and weighted-value intermediates.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _gated_segment_forward(
    logits_pointer,
    transformed_pointer,
    source_pointer,
    destination_pointer,
    row_mask_pointer,
    numerator_pointer,
    denominator_pointer,
    rows,
    dimension: tl.constexpr,
    block_rows: tl.constexpr,
    block_dimension: tl.constexpr,
):
    row_ids = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    columns = (
        tl.program_id(1) * block_dimension
        + tl.arange(0, block_dimension)
    )
    valid_row = row_ids < rows
    active_row = valid_row & (
        tl.load(row_mask_pointer + row_ids, mask=valid_row, other=0.0)
        != 0.0
    )
    mask = active_row[:, None] & (columns[None, :] < dimension)
    source = tl.load(source_pointer + row_ids, mask=active_row, other=0)
    destination = tl.load(
        destination_pointer + row_ids,
        mask=active_row,
        other=0,
    )
    row_offsets = row_ids[:, None] * dimension + columns[None, :]
    source_offsets = source[:, None] * dimension + columns[None, :]
    logits = tl.load(logits_pointer + row_offsets, mask=mask, other=0.0)
    sigma = tl.sigmoid(logits)
    transformed = tl.load(
        transformed_pointer + source_offsets,
        mask=mask,
        other=0.0,
    )
    output_offsets = (
        destination[:, None] * dimension + columns[None, :]
    )
    tl.atomic_add(
        numerator_pointer + output_offsets,
        sigma * transformed,
        mask=mask,
        sem="relaxed",
    )
    tl.atomic_add(
        denominator_pointer + output_offsets,
        sigma,
        mask=mask,
        sem="relaxed",
    )


@triton.jit
def _gated_segment_backward(
    logits_pointer,
    transformed_pointer,
    source_pointer,
    destination_pointer,
    row_mask_pointer,
    grad_numerator_pointer,
    grad_denominator_pointer,
    grad_logits_pointer,
    grad_transformed_pointer,
    rows,
    dimension: tl.constexpr,
    block_rows: tl.constexpr,
    block_dimension: tl.constexpr,
):
    row_ids = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    columns = (
        tl.program_id(1) * block_dimension
        + tl.arange(0, block_dimension)
    )
    valid_row = row_ids < rows
    active_row = valid_row & (
        tl.load(row_mask_pointer + row_ids, mask=valid_row, other=0.0)
        != 0.0
    )
    column_mask = columns[None, :] < dimension
    mask = active_row[:, None] & column_mask
    source = tl.load(source_pointer + row_ids, mask=active_row, other=0)
    destination = tl.load(
        destination_pointer + row_ids,
        mask=active_row,
        other=0,
    )
    row_offsets = row_ids[:, None] * dimension + columns[None, :]
    source_offsets = source[:, None] * dimension + columns[None, :]
    destination_offsets = (
        destination[:, None] * dimension + columns[None, :]
    )
    logits = tl.load(logits_pointer + row_offsets, mask=mask, other=0.0)
    sigma = tl.sigmoid(logits)
    transformed = tl.load(
        transformed_pointer + source_offsets,
        mask=mask,
        other=0.0,
    )
    grad_numerator = tl.load(
        grad_numerator_pointer + destination_offsets,
        mask=mask,
        other=0.0,
    )
    grad_denominator = tl.load(
        grad_denominator_pointer + destination_offsets,
        mask=mask,
        other=0.0,
    )
    grad_sigma = grad_numerator * transformed + grad_denominator
    grad_logits = grad_sigma * sigma * (1.0 - sigma)
    tl.store(
        grad_logits_pointer + row_offsets,
        tl.where(active_row[:, None], grad_logits, 0.0),
        mask=valid_row[:, None] & column_mask,
    )
    tl.atomic_add(
        grad_transformed_pointer + source_offsets,
        grad_numerator * sigma,
        mask=mask,
        sem="relaxed",
    )


class _GatedSegmentSum(Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        transformed,
        source,
        destination,
        row_mask,
        num_nodes,
    ):
        logits = logits.contiguous()
        transformed = transformed.contiguous()
        source = source.to(dtype=torch.int64).contiguous()
        destination = destination.to(dtype=torch.int64).contiguous()
        row_mask = row_mask.reshape(-1).contiguous()
        rows, dimension = logits.shape
        block_rows = 4
        block_dimension = min(triton.next_power_of_2(dimension), 128)
        numerator = transformed.new_zeros((num_nodes, dimension))
        denominator = logits.new_zeros((num_nodes, dimension))
        grid = (
            triton.cdiv(rows, block_rows),
            triton.cdiv(dimension, block_dimension),
        )
        _gated_segment_forward[grid](
            logits,
            transformed,
            source,
            destination,
            row_mask,
            numerator,
            denominator,
            rows,
            dimension=dimension,
            block_rows=block_rows,
            block_dimension=block_dimension,
            num_warps=4,
        )
        ctx.save_for_backward(
            logits,
            transformed,
            source,
            destination,
            row_mask,
        )
        return numerator, denominator

    @staticmethod
    def backward(ctx, grad_numerator, grad_denominator):
        logits, transformed, source, destination, row_mask = ctx.saved_tensors
        rows, dimension = logits.shape
        block_rows = 4
        block_dimension = min(triton.next_power_of_2(dimension), 128)
        grad_logits = torch.empty_like(logits)
        grad_transformed = torch.zeros_like(transformed)
        grid = (
            triton.cdiv(rows, block_rows),
            triton.cdiv(dimension, block_dimension),
        )
        _gated_segment_backward[grid](
            logits,
            transformed,
            source,
            destination,
            row_mask,
            grad_numerator.contiguous(),
            grad_denominator.contiguous(),
            grad_logits,
            grad_transformed,
            rows,
            dimension=dimension,
            block_rows=block_rows,
            block_dimension=block_dimension,
            num_warps=4,
        )
        return grad_logits, grad_transformed, None, None, None, None


def gated_segment_sum(
    logits: torch.Tensor,
    transformed: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    row_mask: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply sigmoid gates and return paired destination reductions."""
    if logits.is_cuda and transformed.is_cuda:
        return _GatedSegmentSum.apply(
            logits,
            transformed,
            source,
            destination,
            row_mask,
            int(num_nodes),
        )
    sigma = torch.sigmoid(logits) * row_mask
    gathered = transformed[source]
    numerator = transformed.new_zeros((int(num_nodes), transformed.shape[1]))
    numerator.index_add_(0, destination, sigma * gathered)
    denominator = sigma.new_zeros((int(num_nodes), sigma.shape[1]))
    denominator.index_add_(0, destination, sigma)
    return numerator, denominator
