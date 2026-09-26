"""Fused GatedMLP epilogue for frozen-parameter inference.

Computes:
    silu(layer_norm(core)) * sigmoid(layer_norm(gate))

The backward returns gradients for the core/gate inputs only. The public wrapper
falls back when LayerNorm parameters require gradients, preserving training
semantics.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _gated_ln_fwd(
    core_x,
    gate_x,
    core_w,
    core_b,
    gate_w,
    gate_b,
    residual,
    res_weight,
    out,
    core_mean,
    core_rstd,
    gate_mean,
    gate_rstd,
    D: tl.constexpr,
    BLK: tl.constexpr,
    EPS_CORE: tl.constexpr,
    EPS_GATE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLK)
    mask = c < D

    core_val = tl.load(core_x + row * D + c, mask=mask, other=0.0).to(tl.float32)
    gate_val = tl.load(gate_x + row * D + c, mask=mask, other=0.0).to(tl.float32)

    c_mu = tl.sum(core_val, axis=0) / D
    c_centered = tl.where(mask, core_val - c_mu, 0.0)
    c_rs = 1.0 / tl.sqrt(tl.sum(c_centered * c_centered, axis=0) / D + EPS_CORE)
    c_y = (
        c_centered
        * c_rs
        * tl.load(core_w + c, mask=mask, other=0.0)
        + tl.load(core_b + c, mask=mask, other=0.0)
    )
    c_sig = tl.sigmoid(c_y)
    core = c_y * c_sig

    g_mu = tl.sum(gate_val, axis=0) / D
    g_centered = tl.where(mask, gate_val - g_mu, 0.0)
    g_rs = 1.0 / tl.sqrt(tl.sum(g_centered * g_centered, axis=0) / D + EPS_GATE)
    g_y = (
        g_centered
        * g_rs
        * tl.load(gate_w + c, mask=mask, other=0.0)
        + tl.load(gate_b + c, mask=mask, other=0.0)
    )
    gate = tl.sigmoid(g_y)

    result = core * gate
    if HAS_RESIDUAL:
        r = tl.load(residual + row * D + c, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(res_weight + c, mask=mask, other=0.0).to(tl.float32)
        result += r * w
    tl.store(out + row * D + c, result, mask=mask)
    tl.store(core_mean + row, c_mu)
    tl.store(core_rstd + row, c_rs)
    tl.store(gate_mean + row, g_mu)
    tl.store(gate_rstd + row, g_rs)


@triton.jit
def _gated_ln_bwd(
    core_x,
    gate_x,
    core_w,
    core_b,
    gate_w,
    gate_b,
    res_weight,
    grad_out,
    core_mean,
    core_rstd,
    gate_mean,
    gate_rstd,
    grad_core_x,
    grad_gate_x,
    grad_residual,
    D: tl.constexpr,
    BLK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLK)
    mask = c < D

    core_val = tl.load(core_x + row * D + c, mask=mask, other=0.0).to(tl.float32)
    gate_val = tl.load(gate_x + row * D + c, mask=mask, other=0.0).to(tl.float32)
    core_w_val = tl.load(core_w + c, mask=mask, other=0.0)
    gate_w_val = tl.load(gate_w + c, mask=mask, other=0.0)
    core_b_val = tl.load(core_b + c, mask=mask, other=0.0)
    gate_b_val = tl.load(gate_b + c, mask=mask, other=0.0)
    grad = tl.load(grad_out + row * D + c, mask=mask, other=0.0).to(tl.float32)

    c_mu = tl.load(core_mean + row)
    c_rs = tl.load(core_rstd + row)
    c_xhat = (core_val - c_mu) * c_rs
    c_y = c_xhat * core_w_val + core_b_val
    c_sig = tl.sigmoid(c_y)
    core = c_y * c_sig

    g_mu = tl.load(gate_mean + row)
    g_rs = tl.load(gate_rstd + row)
    g_xhat = (gate_val - g_mu) * g_rs
    g_y = g_xhat * gate_w_val + gate_b_val
    gate = tl.sigmoid(g_y)

    d_core_y = grad * gate * (c_sig * (1.0 + c_y * (1.0 - c_sig)))
    d_gate_y = grad * core * (gate * (1.0 - gate))

    d_core_xhat = d_core_y * core_w_val
    c_mean_dxhat = tl.sum(tl.where(mask, d_core_xhat, 0.0), axis=0) / D
    c_mean_dxhat_xhat = tl.sum(tl.where(mask, d_core_xhat * c_xhat, 0.0), axis=0) / D
    tl.store(
        grad_core_x + row * D + c,
        (d_core_xhat - c_mean_dxhat - c_xhat * c_mean_dxhat_xhat) * c_rs,
        mask=mask,
    )

    d_gate_xhat = d_gate_y * gate_w_val
    g_mean_dxhat = tl.sum(tl.where(mask, d_gate_xhat, 0.0), axis=0) / D
    g_mean_dxhat_xhat = tl.sum(tl.where(mask, d_gate_xhat * g_xhat, 0.0), axis=0) / D
    tl.store(
        grad_gate_x + row * D + c,
        (d_gate_xhat - g_mean_dxhat - g_xhat * g_mean_dxhat_xhat) * g_rs,
        mask=mask,
    )
    if HAS_RESIDUAL:
        w = tl.load(res_weight + c, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_residual + row * D + c, grad * w, mask=mask)


class _GatedLN(Function):
    @staticmethod
    def forward(ctx, core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate):
        core_x = core_x.contiguous()
        gate_x = gate_x.contiguous()
        rows, dim = core_x.shape
        block = triton.next_power_of_2(dim)
        out = torch.empty_like(core_x)
        core_mean = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        core_rstd = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        gate_mean = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        gate_rstd = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        _gated_ln_fwd[(rows,)](
            core_x,
            gate_x,
            core_w.contiguous(),
            core_b.contiguous(),
            gate_w.contiguous(),
            gate_b.contiguous(),
            core_x,
            core_w,
            out,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
            D=dim,
            BLK=block,
            EPS_CORE=eps_core,
            EPS_GATE=eps_gate,
            HAS_RESIDUAL=False,
            num_warps=1,
        )
        ctx.save_for_backward(
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
        ) = ctx.saved_tensors
        rows, dim = core_x.shape
        block = triton.next_power_of_2(dim)
        grad_core_x = torch.empty_like(core_x)
        grad_gate_x = torch.empty_like(gate_x)
        _gated_ln_bwd[(rows,)](
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            core_w,
            grad_out.contiguous(),
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
            grad_core_x,
            grad_gate_x,
            grad_core_x,
            D=dim,
            BLK=block,
            HAS_RESIDUAL=False,
            num_warps=1,
        )
        return grad_core_x, grad_gate_x, None, None, None, None, None, None


class _GatedLNResidual(Function):
    @staticmethod
    def forward(
        ctx,
        core_x,
        gate_x,
        core_w,
        core_b,
        gate_w,
        gate_b,
        residual,
        res_weight,
        eps_core,
        eps_gate,
    ):
        core_x = core_x.contiguous()
        gate_x = gate_x.contiguous()
        residual = residual.contiguous()
        res_weight = res_weight.reshape(-1).contiguous()
        rows, dim = core_x.shape
        block = triton.next_power_of_2(dim)
        out = torch.empty_like(core_x)
        core_mean = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        core_rstd = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        gate_mean = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        gate_rstd = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        _gated_ln_fwd[(rows,)](
            core_x,
            gate_x,
            core_w.contiguous(),
            core_b.contiguous(),
            gate_w.contiguous(),
            gate_b.contiguous(),
            residual,
            res_weight,
            out,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
            D=dim,
            BLK=block,
            EPS_CORE=eps_core,
            EPS_GATE=eps_gate,
            HAS_RESIDUAL=True,
            num_warps=1,
        )
        ctx.save_for_backward(
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            res_weight,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            res_weight,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
        ) = ctx.saved_tensors
        rows, dim = core_x.shape
        block = triton.next_power_of_2(dim)
        grad_core_x = torch.empty_like(core_x)
        grad_gate_x = torch.empty_like(gate_x)
        grad_residual = torch.empty_like(core_x)
        _gated_ln_bwd[(rows,)](
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            res_weight,
            grad_out.contiguous(),
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
            grad_core_x,
            grad_gate_x,
            grad_residual,
            D=dim,
            BLK=block,
            HAS_RESIDUAL=True,
            num_warps=1,
        )
        return (
            grad_core_x,
            grad_gate_x,
            None,
            None,
            None,
            None,
            grad_residual,
            None,
            None,
            None,
        )


def _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate):
    from .triton_layernorm_act import ACT_SIGMOID, ACT_SILU, fused_ln_act

    core = fused_ln_act(core_x, core_w, core_b, eps_core, ACT_SILU)
    gate = fused_ln_act(gate_x, gate_w, gate_b, eps_gate, ACT_SIGMOID)
    return core * gate


def fused_gated_ln(
    core_x: torch.Tensor,
    gate_x: torch.Tensor,
    core_w: torch.Tensor,
    core_b: torch.Tensor,
    gate_w: torch.Tensor,
    gate_b: torch.Tensor,
    eps_core: float = 1e-5,
    eps_gate: float = 1e-5,
) -> torch.Tensor:
    if not (core_x.is_cuda and gate_x.is_cuda):
        return _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate)
    if core_x.shape != gate_x.shape:
        return _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate)
    if core_w.requires_grad or core_b.requires_grad or gate_w.requires_grad or gate_b.requires_grad:
        return _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate)
    return _GatedLN.apply(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate)


def fused_gated_ln_residual(
    core_x: torch.Tensor,
    gate_x: torch.Tensor,
    core_w: torch.Tensor,
    core_b: torch.Tensor,
    gate_w: torch.Tensor,
    gate_b: torch.Tensor,
    residual: torch.Tensor,
    res_weight: torch.Tensor,
    eps_core: float = 1e-5,
    eps_gate: float = 1e-5,
) -> torch.Tensor:
    if not (core_x.is_cuda and gate_x.is_cuda and residual.is_cuda and res_weight.is_cuda):
        return (
            _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate)
            + res_weight * residual
        )
    if core_x.shape != gate_x.shape or residual.shape != core_x.shape:
        return (
            _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate)
            + res_weight * residual
        )
    if core_x.shape[1] != res_weight.numel():
        return (
            _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate)
            + res_weight * residual
        )
    if (
        core_w.requires_grad
        or core_b.requires_grad
        or gate_w.requires_grad
        or gate_b.requires_grad
        or res_weight.requires_grad
    ):
        return (
            _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate)
            + res_weight * residual
        )
    return _GatedLNResidual.apply(
        core_x,
        gate_x,
        core_w,
        core_b,
        gate_w,
        gate_b,
        residual,
        res_weight,
        eps_core,
        eps_gate,
    )
