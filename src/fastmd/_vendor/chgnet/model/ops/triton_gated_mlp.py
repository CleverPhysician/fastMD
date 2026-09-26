"""Fused frozen-inference LayerNorm and gate for CHGNet GatedMLPs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _layer_norm_gate_fwd(
    core,
    gate,
    core_weight,
    core_bias,
    gate_weight,
    gate_bias,
    out,
    core_mean,
    core_rstd,
    gate_mean,
    gate_rstd,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    CORE_STRIDE: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    out_offsets = row * N + cols
    core_offsets = row * CORE_STRIDE + cols
    gate_offsets = row * GATE_STRIDE + cols

    core_value = tl.load(core + core_offsets, mask=mask, other=0.0).to(tl.float32)
    gate_value = tl.load(gate + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    core_mu = tl.sum(core_value, axis=0) / N
    gate_mu = tl.sum(gate_value, axis=0) / N
    core_centered = tl.where(mask, core_value - core_mu, 0.0)
    gate_centered = tl.where(mask, gate_value - gate_mu, 0.0)
    core_var = tl.sum(core_centered * core_centered, axis=0) / N
    gate_var = tl.sum(gate_centered * gate_centered, axis=0) / N
    core_inv_std = tl.rsqrt(core_var + EPS)
    gate_inv_std = tl.rsqrt(gate_var + EPS)

    cw = tl.load(core_weight + cols, mask=mask, other=0.0).to(tl.float32)
    cb = tl.load(core_bias + cols, mask=mask, other=0.0).to(tl.float32)
    gw = tl.load(gate_weight + cols, mask=mask, other=0.0).to(tl.float32)
    gb = tl.load(gate_bias + cols, mask=mask, other=0.0).to(tl.float32)
    core_norm = core_centered * core_inv_std * cw + cb
    gate_norm = gate_centered * gate_inv_std * gw + gb
    core_sigmoid = tl.sigmoid(core_norm)
    activated_core = core_norm * core_sigmoid
    activated_gate = tl.sigmoid(gate_norm)
    tl.store(out + out_offsets, activated_core * activated_gate, mask=mask)
    tl.store(core_mean + row, core_mu)
    tl.store(core_rstd + row, core_inv_std)
    tl.store(gate_mean + row, gate_mu)
    tl.store(gate_rstd + row, gate_inv_std)


@triton.jit
def _layer_norm_gate_bwd(
    grad_out,
    core,
    gate,
    core_weight,
    core_bias,
    gate_weight,
    gate_bias,
    core_mean,
    core_rstd,
    gate_mean,
    gate_rstd,
    grad_core,
    grad_gate,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    CORE_STRIDE: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offsets = row * N + cols
    core_offsets = row * CORE_STRIDE + cols
    gate_offsets = row * GATE_STRIDE + cols

    grad = tl.load(grad_out + offsets, mask=mask, other=0.0).to(tl.float32)
    core_value = tl.load(core + core_offsets, mask=mask, other=0.0).to(tl.float32)
    gate_value = tl.load(gate + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    core_mu = tl.load(core_mean + row)
    core_inv_std = tl.load(core_rstd + row)
    gate_mu = tl.load(gate_mean + row)
    gate_inv_std = tl.load(gate_rstd + row)
    cw = tl.load(core_weight + cols, mask=mask, other=0.0).to(tl.float32)
    cb = tl.load(core_bias + cols, mask=mask, other=0.0).to(tl.float32)
    gw = tl.load(gate_weight + cols, mask=mask, other=0.0).to(tl.float32)
    gb = tl.load(gate_bias + cols, mask=mask, other=0.0).to(tl.float32)

    core_hat = (core_value - core_mu) * core_inv_std
    gate_hat = (gate_value - gate_mu) * gate_inv_std
    core_norm = core_hat * cw + cb
    gate_norm = gate_hat * gw + gb
    core_sigmoid = tl.sigmoid(core_norm)
    activated_core = core_norm * core_sigmoid
    gate_sigmoid = tl.sigmoid(gate_norm)
    d_core_norm = (
        grad
        * gate_sigmoid
        * core_sigmoid
        * (1.0 + core_norm * (1.0 - core_sigmoid))
    )
    d_gate_norm = (
        grad * activated_core * gate_sigmoid * (1.0 - gate_sigmoid)
    )

    d_core_hat = d_core_norm * cw
    d_gate_hat = d_gate_norm * gw
    core_sum = tl.sum(tl.where(mask, d_core_hat, 0.0), axis=0)
    gate_sum = tl.sum(tl.where(mask, d_gate_hat, 0.0), axis=0)
    core_dot = tl.sum(tl.where(mask, d_core_hat * core_hat, 0.0), axis=0)
    gate_dot = tl.sum(tl.where(mask, d_gate_hat * gate_hat, 0.0), axis=0)
    core_grad = core_inv_std * (
        d_core_hat - core_sum / N - core_hat * core_dot / N
    )
    gate_grad = gate_inv_std * (
        d_gate_hat - gate_sum / N - gate_hat * gate_dot / N
    )
    tl.store(grad_core + offsets, core_grad, mask=mask)
    tl.store(grad_gate + offsets, gate_grad, mask=mask)


class _LayerNormSiLUGate(Function):
    @staticmethod
    def forward(
        ctx,
        core,
        gate,
        core_weight,
        core_bias,
        gate_weight,
        gate_bias,
        eps,
    ):
        rows, width = core.shape
        # Kernels below write a compact row-major matrix.  ``core`` can be a
        # column chunk of the packed projection (stride ``2 * width``), so do
        # not inherit its layout here.
        out = torch.empty(core.shape, dtype=core.dtype, device=core.device)
        stats = torch.empty((4, rows), dtype=torch.float32, device=core.device)
        block = triton.next_power_of_2(width)
        _layer_norm_gate_fwd[(rows,)](
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            out,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            N=width,
            BLOCK=block,
            EPS=float(eps),
            CORE_STRIDE=core.stride(0),
            GATE_STRIDE=gate.stride(0),
            num_warps=4,
        )
        ctx.save_for_backward(
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            stats,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            stats,
        ) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_core = torch.empty(core.shape, dtype=core.dtype, device=core.device)
        grad_gate = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
        rows, width = core.shape
        block = triton.next_power_of_2(width)
        _layer_norm_gate_bwd[(rows,)](
            grad_out,
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            grad_core,
            grad_gate,
            N=width,
            BLOCK=block,
            CORE_STRIDE=core.stride(0),
            GATE_STRIDE=gate.stride(0),
            num_warps=4,
        )
        return grad_core, grad_gate, None, None, None, None, None


def layer_norm_silu_gate(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_weight: torch.Tensor,
    core_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply two LayerNorms and the CHGNet ``SiLU(core) * sigmoid(gate)``."""
    if core.ndim != 2 or gate.shape != core.shape:
        raise ValueError("core and gate must be equal rank-2 matrices")
    if not core.is_floating_point() or core.dtype != gate.dtype:
        raise TypeError("core and gate must use the same floating-point dtype")
    width = core.shape[1]
    affine = (core_weight, core_bias, gate_weight, gate_bias)
    if any(value.shape != (width,) for value in affine):
        raise ValueError("LayerNorm affine tensors must match the feature width")
    if any(value.device != core.device for value in (*affine, gate)):
        raise ValueError("all fused gate inputs must be on the same device")
    if any(value.dtype != core.dtype for value in affine):
        raise TypeError("all fused gate inputs must use the same dtype")
    if eps <= 0:
        raise ValueError("LayerNorm epsilon must be positive")
    use_triton = (
        core.is_cuda
        and core.dtype == torch.float32
        and core.shape[0] > 0
        and 0 < width <= 256
        and core.stride(1) == 1
        and gate.stride(1) == 1
        and all(value.stride(0) == 1 for value in affine)
        and not any(value.requires_grad for value in affine)
    )
    if use_triton:
        return _LayerNormSiLUGate.apply(
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            float(eps),
        )
    core_norm = F.layer_norm(
        core,
        (width,),
        core_weight,
        core_bias,
        eps,
    )
    gate_norm = F.layer_norm(
        gate,
        (width,),
        gate_weight,
        gate_bias,
        eps,
    )
    return F.silu(core_norm) * torch.sigmoid(gate_norm)
