"""Fused line-graph envelope gather product.

Computes ``out[row, d] = base[source[row], d] * base[target[row], d]`` and
fuses the two index_select calls plus the multiply in the line-graph refinement
envelope path.
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


_BM = _env_int("MATRIS_LINE_ENVELOPE_BM", 4)
_BD = _env_int("MATRIS_LINE_ENVELOPE_BD", 128)


@triton.jit
def _line_envelope_fwd(
    base,
    source,
    target,
    out,
    N: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    src = tl.load(source + rows, mask=rows < N, other=0)
    tgt = tl.load(target + rows, mask=rows < N, other=0)
    a = tl.load(base + src[:, None] * D + cols[None, :], mask=mask, other=0.0)
    b = tl.load(base + tgt[:, None] * D + cols[None, :], mask=mask, other=0.0)
    tl.store(out + rows[:, None] * D + cols[None, :], a * b, mask=mask)


@triton.jit
def _line_envelope_bwd(
    gout,
    base,
    source,
    target,
    grad_base,
    N: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    src = tl.load(source + rows, mask=rows < N, other=0)
    tgt = tl.load(target + rows, mask=rows < N, other=0)
    offs = rows[:, None] * D + cols[None, :]
    go = tl.load(gout + offs, mask=mask, other=0.0)
    a = tl.load(base + src[:, None] * D + cols[None, :], mask=mask, other=0.0)
    b = tl.load(base + tgt[:, None] * D + cols[None, :], mask=mask, other=0.0)
    tl.atomic_add(grad_base + src[:, None] * D + cols[None, :], go * b, mask=mask, sem="relaxed")
    tl.atomic_add(grad_base + tgt[:, None] * D + cols[None, :], go * a, mask=mask, sem="relaxed")


class _LineEnvelopeProduct(Function):
    @staticmethod
    def forward(ctx, base: torch.Tensor, source: torch.Tensor, target: torch.Tensor):
        if base.dim() != 2:
            raise ValueError(f"expected base [S,D], got {base.shape}")
        base = base.contiguous() if base.stride(1) != 1 else base
        source = source.contiguous()
        target = target.contiguous()
        n = source.numel()
        d = base.shape[1]
        out = torch.empty((n, d), device=base.device, dtype=base.dtype)
        bd = triton.next_power_of_2(min(_BD, d))
        grid = (triton.cdiv(n, _BM), triton.cdiv(d, _BD))
        _line_envelope_fwd[grid](
            base,
            source,
            target,
            out,
            N=n,
            D=d,
            BM=_BM,
            BD=bd,
            num_warps=4,
        )
        ctx.save_for_backward(base, source, target)
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        base, source, target = ctx.saved_tensors
        gout = gout.contiguous()
        n, d = gout.shape
        grad_base = torch.zeros_like(base)
        bd = triton.next_power_of_2(min(_BD, d))
        grid = (triton.cdiv(n, _BM), triton.cdiv(d, _BD))
        _line_envelope_bwd[grid](
            gout,
            base,
            source,
            target,
            grad_base,
            N=n,
            D=d,
            BM=_BM,
            BD=bd,
            num_warps=4,
        )
        return grad_base, None, None


def line_envelope_product(
    base: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if base.is_cuda and source.is_cuda and target.is_cuda:
        return _LineEnvelopeProduct.apply(base, source, target)
    return torch.index_select(base, 0, source) * torch.index_select(base, 0, target)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for n, s, d in [(1024, 113, 128), (152446, 13087, 128)]:
        source = torch.randint(0, s, (n,), device=dev)
        target = torch.randint(0, s, (n,), device=dev)
        base = torch.randn(s, d, device=dev, requires_grad=True)
        base_tri = base.detach().clone().requires_grad_(True)
        ref = torch.index_select(base, 0, source) * torch.index_select(base, 0, target)
        tri = line_envelope_product(base_tri, source, target)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        print(
            f"N={n} S={s} D={d}: "
            f"fwd={(ref - tri).abs().max().item():.2e} "
            f"bwd={(base.grad - base_tri.grad).abs().max().item():.2e}"
        )
