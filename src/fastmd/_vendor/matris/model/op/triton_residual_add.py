"""Fused residual add for frozen-parameter inference.

Computes ``delta + weight * residual`` for 2D feature tensors.  The wrapper in
``interaction_block`` only uses this when ``weight`` is frozen; backward returns
input gradients for ``delta`` and ``residual`` and no parameter gradient.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch.autograd import Function


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


_BM = _env_int("MATRIS_FUSED_RESIDUAL_ADD_BM", 4)
_BD = _env_int("MATRIS_FUSED_RESIDUAL_ADD_BD", 128)


@triton.jit
def _residual_add_fwd(
    delta,
    residual,
    weight,
    out,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    offs = rows[:, None] * D + cols[None, :]
    d = tl.load(delta + offs, mask=mask, other=0.0)
    r = tl.load(residual + offs, mask=mask, other=0.0)
    w = tl.load(weight + cols, mask=cols < D, other=0.0)
    tl.store(out + offs, d + r * w[None, :], mask=mask)


@triton.jit
def _residual_add_bwd(
    gout,
    weight,
    grad_residual,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    offs = rows[:, None] * D + cols[None, :]
    go = tl.load(gout + offs, mask=mask, other=0.0)
    w = tl.load(weight + cols, mask=cols < D, other=0.0)
    tl.store(grad_residual + offs, go * w[None, :], mask=mask)


class _ResidualAdd(Function):
    @staticmethod
    def forward(ctx, delta: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
        delta = delta.contiguous()
        residual = residual.contiguous()
        weight = weight.reshape(-1).contiguous()
        n, d = delta.shape
        out = torch.empty_like(delta)
        bd = triton.next_power_of_2(min(_BD, d))
        grid = (triton.cdiv(n, _BM), triton.cdiv(d, bd))
        _residual_add_fwd[grid](
            delta,
            residual,
            weight,
            out,
            N=n,
            D=d,
            BM=_BM,
            BD=bd,
            num_warps=4,
        )
        ctx.save_for_backward(weight)
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (weight,) = ctx.saved_tensors
        gout = gout.contiguous()
        n, d = gout.shape
        grad_residual = torch.empty_like(gout)
        bd = triton.next_power_of_2(min(_BD, d))
        grid = (triton.cdiv(n, _BM), triton.cdiv(d, bd))
        _residual_add_bwd[grid](
            gout,
            weight,
            grad_residual,
            N=n,
            D=d,
            BM=_BM,
            BD=bd,
            num_warps=4,
        )
        return gout, grad_residual, None


def residual_add(
    delta: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return _ResidualAdd.apply(delta, residual, weight)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for n, d in [(1024, 128), (13824, 128), (155648, 128)]:
        delta = torch.randn(n, d, device=dev, requires_grad=True)
        residual = torch.randn(n, d, device=dev, requires_grad=True)
        weight = torch.randn(1, d, device=dev)
        d2 = delta.detach().clone().requires_grad_(True)
        r2 = residual.detach().clone().requires_grad_(True)
        ref = delta + weight * residual
        tri = residual_add(d2, r2, weight)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        torch.cuda.synchronize()
        print(
            f"N={n}: fwd={(ref - tri).abs().max().item():.2e} "
            f"bwd={max((delta.grad - d2.grad).abs().max().item(), (residual.grad - r2.grad).abs().max().item()):.2e}"
        )
