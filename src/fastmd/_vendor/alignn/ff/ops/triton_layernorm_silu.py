"""Triton fusion for frozen LayerNorm followed by SiLU.

This is the ALIGNN specialization of MatRIS opt commit ``ec1727f``'s generic
LayerNorm/activation fusion. Model parameters are frozen during MD, so the
custom backward only needs the input gradient used to obtain atomic forces.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function
from torch.nn import functional


@triton.jit
def _layernorm_silu_forward(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    mean_pointer,
    reciprocal_std_pointer,
    residual_pointer,
    row_mask_pointer,
    dimension: tl.constexpr,
    block: tl.constexpr,
    epsilon: tl.constexpr,
    has_residual: tl.constexpr,
    has_row_mask: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    column_mask = columns < dimension
    row_active = True
    if has_row_mask:
        row_active = tl.load(row_mask_pointer + row) != 0.0
    mask = column_mask & row_active
    values = tl.load(
        input_pointer + row * dimension + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / dimension
    centered = tl.where(column_mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / dimension
    reciprocal_std = 1.0 / tl.sqrt(variance + epsilon)
    normalized = centered * reciprocal_std
    affine = (
        normalized
        * tl.load(weight_pointer + columns, mask=mask, other=0.0)
        + tl.load(bias_pointer + columns, mask=mask, other=0.0)
    )
    output = affine * tl.sigmoid(affine)
    if has_residual:
        output += tl.load(
            residual_pointer + row * dimension + columns,
            mask=mask,
            other=0.0,
        )
    if has_row_mask:
        output = tl.where(row_active, output, 0.0)
    tl.store(
        output_pointer + row * dimension + columns,
        output,
        mask=column_mask,
    )
    tl.store(mean_pointer + row, tl.where(row_active, mean, 0.0))
    tl.store(
        reciprocal_std_pointer + row,
        tl.where(row_active, reciprocal_std, 0.0),
    )


@triton.jit
def _layernorm_silu_backward(
    input_pointer,
    weight_pointer,
    bias_pointer,
    grad_output_pointer,
    mean_pointer,
    reciprocal_std_pointer,
    grad_input_pointer,
    grad_residual_pointer,
    row_mask_pointer,
    dimension: tl.constexpr,
    block: tl.constexpr,
    has_residual: tl.constexpr,
    has_row_mask: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    column_mask = columns < dimension
    row_active = True
    if has_row_mask:
        row_active = tl.load(row_mask_pointer + row) != 0.0
    mask = column_mask & row_active
    values = tl.load(
        input_pointer + row * dimension + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(mean_pointer + row, mask=row_active, other=0.0)
    reciprocal_std = tl.load(
        reciprocal_std_pointer + row,
        mask=row_active,
        other=0.0,
    )
    normalized = (values - mean) * reciprocal_std
    weight = tl.load(weight_pointer + columns, mask=mask, other=0.0)
    affine = weight * normalized + tl.load(
        bias_pointer + columns,
        mask=mask,
        other=0.0,
    )
    grad_output = tl.load(
        grad_output_pointer + row * dimension + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if has_residual:
        tl.store(
            grad_residual_pointer + row * dimension + columns,
            tl.where(row_active, grad_output, 0.0),
            mask=column_mask,
        )
    sigmoid = tl.sigmoid(affine)
    grad_affine = grad_output * (
        sigmoid * (1.0 + affine * (1.0 - sigmoid))
    )
    grad_normalized = grad_affine * weight
    mean_grad = (
        tl.sum(tl.where(mask, grad_normalized, 0.0), axis=0) / dimension
    )
    mean_grad_normalized = (
        tl.sum(
            tl.where(mask, grad_normalized * normalized, 0.0), axis=0
        )
        / dimension
    )
    grad_input = (
        grad_normalized
        - mean_grad
        - normalized * mean_grad_normalized
    ) * reciprocal_std
    tl.store(
        grad_input_pointer + row * dimension + columns,
        grad_input,
        mask=column_mask,
    )


class _LayerNormSiLU(Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        weight,
        bias,
        residual,
        row_mask,
        epsilon,
        has_residual,
        has_row_mask,
    ):
        inputs = inputs.contiguous()
        rows, dimension = inputs.shape
        block = triton.next_power_of_2(dimension)
        output = torch.empty_like(inputs)
        mean = torch.empty(rows, device=inputs.device, dtype=torch.float32)
        reciprocal_std = torch.empty_like(mean)
        _layernorm_silu_forward[(rows,)](
            inputs,
            weight,
            bias,
            output,
            mean,
            reciprocal_std,
            residual,
            row_mask,
            dimension=dimension,
            block=block,
            epsilon=epsilon,
            has_residual=has_residual,
            has_row_mask=has_row_mask,
        )
        ctx.save_for_backward(
            inputs,
            weight,
            bias,
            mean,
            reciprocal_std,
            row_mask,
        )
        ctx.has_residual = has_residual
        ctx.has_row_mask = has_row_mask
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias, mean, reciprocal_std, row_mask = (
            ctx.saved_tensors
        )
        rows, dimension = inputs.shape
        block = triton.next_power_of_2(dimension)
        grad_input = torch.empty_like(inputs)
        grad_residual = (
            torch.empty_like(inputs) if ctx.has_residual else inputs
        )
        _layernorm_silu_backward[(rows,)](
            inputs,
            weight,
            bias,
            grad_output.contiguous(),
            mean,
            reciprocal_std,
            grad_input,
            grad_residual,
            row_mask,
            dimension=dimension,
            block=block,
            has_residual=ctx.has_residual,
            has_row_mask=ctx.has_row_mask,
        )
        return (
            grad_input,
            None,
            None,
            grad_residual if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def fused_layernorm_silu(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse LayerNorm, SiLU, optional residual, and optional row mask."""
    residual_ok = residual is None or (
        residual.is_cuda and residual.shape == inputs.shape
    )
    mask_ok = row_mask is None or (
        row_mask.is_cuda
        and row_mask.numel() == inputs.shape[0]
        and row_mask.dtype == inputs.dtype
    )
    can_fuse = (
        inputs.is_cuda
        and inputs.ndim == 2
        and inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not weight.requires_grad
        and not bias.requires_grad
        and residual_ok
        and mask_ok
    )
    if can_fuse:
        residual_buffer = inputs if residual is None else residual
        mask_buffer = inputs if row_mask is None else row_mask
        return _LayerNormSiLU.apply(
            inputs,
            weight.contiguous(),
            bias.contiguous(),
            residual_buffer.contiguous(),
            mask_buffer.contiguous(),
            float(epsilon),
            residual is not None,
            row_mask is not None,
        )
    normalized = functional.layer_norm(
        inputs,
        (inputs.shape[-1],),
        weight,
        bias,
        epsilon,
    )
    output = functional.silu(normalized)
    if residual is not None:
        output = output + residual
    if row_mask is not None:
        output = output * row_mask
    return output
