"""Fused line-graph angle construction and Gaussian RBF expansion.

This specializes MatRIS opt commit ``acc7808``'s three-body basis kernel for
ALIGNN, whose angle embedding uses Gaussian radial basis functions of the bond
cosine rather than a Fourier basis of the angle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _angle_rbf_forward(
    displacement_pointer,
    line_source_pointer,
    line_destination_pointer,
    centers_pointer,
    row_mask_pointer,
    output_pointer,
    rows,
    bins: tl.constexpr,
    block: tl.constexpr,
    gamma: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    column_mask = columns < bins
    valid_row = row < rows
    row_active = valid_row & (
        tl.load(row_mask_pointer + row, mask=valid_row, other=0.0) != 0.0
    )
    source = tl.load(line_source_pointer + row, mask=row_active, other=0)
    destination = tl.load(
        line_destination_pointer + row,
        mask=row_active,
        other=0,
    )
    source_offset = source * 3
    destination_offset = destination * 3
    r10 = -tl.load(
        displacement_pointer + source_offset,
        mask=row_active,
        other=0.0,
    )
    r11 = -tl.load(
        displacement_pointer + source_offset + 1,
        mask=row_active,
        other=0.0,
    )
    r12 = -tl.load(
        displacement_pointer + source_offset + 2,
        mask=row_active,
        other=0.0,
    )
    r20 = tl.load(
        displacement_pointer + destination_offset,
        mask=row_active,
        other=0.0,
    )
    r21 = tl.load(
        displacement_pointer + destination_offset + 1,
        mask=row_active,
        other=0.0,
    )
    r22 = tl.load(
        displacement_pointer + destination_offset + 2,
        mask=row_active,
        other=0.0,
    )
    norm1_squared = r10 * r10 + r11 * r11 + r12 * r12
    norm2_squared = r20 * r20 + r21 * r21 + r22 * r22
    denominator = tl.sqrt(norm1_squared * norm2_squared)
    denominator = tl.maximum(denominator, 1.0e-12)
    raw_cosine = (r10 * r20 + r11 * r21 + r12 * r22) / denominator
    cosine = tl.minimum(1.0, tl.maximum(-1.0, raw_cosine))
    centers = tl.load(
        centers_pointer + columns,
        mask=column_mask & row_active,
        other=0.0,
    )
    delta = cosine - centers
    basis = tl.exp(-gamma * delta * delta)
    basis = tl.where(row_active, basis, 0.0)
    tl.store(
        output_pointer + row * bins + columns,
        basis,
        mask=valid_row & column_mask,
    )


@triton.jit
def _angle_rbf_backward(
    displacement_pointer,
    line_source_pointer,
    line_destination_pointer,
    centers_pointer,
    row_mask_pointer,
    grad_output_pointer,
    grad_displacement_pointer,
    rows,
    bins: tl.constexpr,
    block: tl.constexpr,
    gamma: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    column_mask = columns < bins
    valid_row = row < rows
    row_active = valid_row & (
        tl.load(row_mask_pointer + row, mask=valid_row, other=0.0) != 0.0
    )
    source = tl.load(line_source_pointer + row, mask=row_active, other=0)
    destination = tl.load(
        line_destination_pointer + row,
        mask=row_active,
        other=0,
    )
    source_offset = source * 3
    destination_offset = destination * 3
    r10 = -tl.load(
        displacement_pointer + source_offset,
        mask=row_active,
        other=0.0,
    )
    r11 = -tl.load(
        displacement_pointer + source_offset + 1,
        mask=row_active,
        other=0.0,
    )
    r12 = -tl.load(
        displacement_pointer + source_offset + 2,
        mask=row_active,
        other=0.0,
    )
    r20 = tl.load(
        displacement_pointer + destination_offset,
        mask=row_active,
        other=0.0,
    )
    r21 = tl.load(
        displacement_pointer + destination_offset + 1,
        mask=row_active,
        other=0.0,
    )
    r22 = tl.load(
        displacement_pointer + destination_offset + 2,
        mask=row_active,
        other=0.0,
    )
    norm1_squared = tl.maximum(
        r10 * r10 + r11 * r11 + r12 * r12,
        1.0e-12,
    )
    norm2_squared = tl.maximum(
        r20 * r20 + r21 * r21 + r22 * r22,
        1.0e-12,
    )
    denominator = tl.sqrt(norm1_squared * norm2_squared)
    raw_cosine = (r10 * r20 + r11 * r21 + r12 * r22) / denominator
    cosine = tl.minimum(1.0, tl.maximum(-1.0, raw_cosine))
    centers = tl.load(
        centers_pointer + columns,
        mask=column_mask & row_active,
        other=0.0,
    )
    delta = cosine - centers
    basis = tl.exp(-gamma * delta * delta)
    grad_output = tl.load(
        grad_output_pointer + row * bins + columns,
        mask=column_mask & row_active,
        other=0.0,
    )
    grad_cosine = tl.sum(
        tl.where(
            column_mask & row_active,
            grad_output * basis * (-2.0 * gamma * delta),
            0.0,
        ),
        axis=0,
    )
    inside_clamp = (raw_cosine >= -1.0) & (raw_cosine <= 1.0)
    grad_cosine = tl.where(inside_clamp, grad_cosine, 0.0)

    grad_r10 = grad_cosine * (
        r20 / denominator - cosine * r10 / norm1_squared
    )
    grad_r11 = grad_cosine * (
        r21 / denominator - cosine * r11 / norm1_squared
    )
    grad_r12 = grad_cosine * (
        r22 / denominator - cosine * r12 / norm1_squared
    )
    grad_r20 = grad_cosine * (
        r10 / denominator - cosine * r20 / norm2_squared
    )
    grad_r21 = grad_cosine * (
        r11 / denominator - cosine * r21 / norm2_squared
    )
    grad_r22 = grad_cosine * (
        r12 / denominator - cosine * r22 / norm2_squared
    )
    tl.atomic_add(
        grad_displacement_pointer + source_offset,
        -grad_r10,
        mask=row_active,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_displacement_pointer + source_offset + 1,
        -grad_r11,
        mask=row_active,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_displacement_pointer + source_offset + 2,
        -grad_r12,
        mask=row_active,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_displacement_pointer + destination_offset,
        grad_r20,
        mask=row_active,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_displacement_pointer + destination_offset + 1,
        grad_r21,
        mask=row_active,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_displacement_pointer + destination_offset + 2,
        grad_r22,
        mask=row_active,
        sem="relaxed",
    )


class _AngleRBF(Function):
    @staticmethod
    def forward(
        ctx,
        displacement,
        line_source,
        line_destination,
        centers,
        gamma,
        row_mask,
    ):
        displacement = displacement.contiguous()
        line_source = line_source.to(dtype=torch.int64).contiguous()
        line_destination = line_destination.to(dtype=torch.int64).contiguous()
        centers = centers.contiguous()
        row_mask = row_mask.reshape(-1).contiguous()
        rows = line_source.numel()
        bins = centers.numel()
        block = triton.next_power_of_2(bins)
        output = displacement.new_empty((rows, bins))
        _angle_rbf_forward[(rows,)](
            displacement,
            line_source,
            line_destination,
            centers,
            row_mask,
            output,
            rows,
            bins=bins,
            block=block,
            gamma=float(gamma),
            num_warps=1,
        )
        ctx.save_for_backward(
            displacement,
            line_source,
            line_destination,
            centers,
            row_mask,
        )
        ctx.gamma = float(gamma)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            displacement,
            line_source,
            line_destination,
            centers,
            row_mask,
        ) = ctx.saved_tensors
        rows = line_source.numel()
        bins = centers.numel()
        block = triton.next_power_of_2(bins)
        grad_displacement = torch.zeros_like(displacement)
        _angle_rbf_backward[(rows,)](
            displacement,
            line_source,
            line_destination,
            centers,
            row_mask,
            grad_output.contiguous(),
            grad_displacement,
            rows,
            bins=bins,
            block=block,
            gamma=ctx.gamma,
            num_warps=1,
        )
        return grad_displacement, None, None, None, None, None


def angle_rbf(
    displacement: torch.Tensor,
    line_source: torch.Tensor,
    line_destination: torch.Tensor,
    centers: torch.Tensor,
    gamma: float,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked Gaussian RBF features for line-graph bond cosines."""
    if displacement.is_cuda:
        return _AngleRBF.apply(
            displacement,
            line_source,
            line_destination,
            centers,
            float(gamma),
            row_mask,
        )
    r1 = -displacement[line_source]
    r2 = displacement[line_destination]
    cosine = torch.sum(r1 * r2, dim=1) / (
        torch.linalg.vector_norm(r1, dim=1)
        * torch.linalg.vector_norm(r2, dim=1)
    )
    cosine = torch.clamp(cosine, -1.0, 1.0)
    basis = torch.exp(
        -float(gamma) * (cosine[:, None] - centers[None, :]) ** 2
    )
    return basis * row_mask
