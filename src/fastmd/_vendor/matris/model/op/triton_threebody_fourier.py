"""Fused Fourier three-body basis for line-graph angles.

Computes the current PyTorch sequence:

    edge_i = unit_edge_vectors[target_index]
    edge_j = unit_edge_vectors[source_index]
    angle = acos(sum(edge_i * edge_j) * (1 - 1e-6))
    basis = [1/sqrt(2), sin(angle * f), cos(angle * f)] / sqrt(pi)

The Fourier frequencies must be frozen; backward returns gradients only for
``unit_edge_vectors``.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch.autograd import Function
from triton.language.extra import libdevice


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


_BM = _env_int("MATRIS_THREEBODY_FOURIER_BM", 128)
_EPS_SCALE = 1.0 - 1.0e-6


@triton.jit
def _threebody_fourier_fwd(
    unit_vec,
    target_index,
    source_index,
    freqs,
    out,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < N
    tgt = tl.load(target_index + rows, mask=mask, other=0)
    src = tl.load(source_index + rows, mask=mask, other=0)

    ti0 = tl.load(unit_vec + tgt * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    ti1 = tl.load(unit_vec + tgt * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    ti2 = tl.load(unit_vec + tgt * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    sj0 = tl.load(unit_vec + src * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    sj1 = tl.load(unit_vec + src * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    sj2 = tl.load(unit_vec + src * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    theta = (ti0 * sj0 + ti1 * sj1 + ti2 * sj2) * 0.999999
    angle = libdevice.acos(theta)
    inv_scale = 1.0 / SCALE
    base = out + rows * 7
    tl.store(base + 0, 0.7071067811865476 * inv_scale, mask=mask)

    f0 = tl.load(freqs + 0).to(tl.float32)
    f1 = tl.load(freqs + 1).to(tl.float32)
    f2 = tl.load(freqs + 2).to(tl.float32)
    a0 = angle * f0
    a1 = angle * f1
    a2 = angle * f2
    tl.store(base + 1, libdevice.sin(a0) * inv_scale, mask=mask)
    tl.store(base + 2, libdevice.sin(a1) * inv_scale, mask=mask)
    tl.store(base + 3, libdevice.sin(a2) * inv_scale, mask=mask)
    tl.store(base + 4, libdevice.cos(a0) * inv_scale, mask=mask)
    tl.store(base + 5, libdevice.cos(a1) * inv_scale, mask=mask)
    tl.store(base + 6, libdevice.cos(a2) * inv_scale, mask=mask)


@triton.jit
def _threebody_fourier_bwd(
    gout,
    unit_vec,
    target_index,
    source_index,
    freqs,
    grad_unit,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < N
    tgt = tl.load(target_index + rows, mask=mask, other=0)
    src = tl.load(source_index + rows, mask=mask, other=0)

    ti0 = tl.load(unit_vec + tgt * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    ti1 = tl.load(unit_vec + tgt * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    ti2 = tl.load(unit_vec + tgt * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    sj0 = tl.load(unit_vec + src * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    sj1 = tl.load(unit_vec + src * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    sj2 = tl.load(unit_vec + src * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    theta = (ti0 * sj0 + ti1 * sj1 + ti2 * sj2) * 0.999999
    angle = libdevice.acos(theta)
    inv_scale = 1.0 / SCALE
    base = gout + rows * 7

    f0 = tl.load(freqs + 0).to(tl.float32)
    f1 = tl.load(freqs + 1).to(tl.float32)
    f2 = tl.load(freqs + 2).to(tl.float32)
    a0 = angle * f0
    a1 = angle * f1
    a2 = angle * f2
    gsin0 = tl.load(base + 1, mask=mask, other=0.0).to(tl.float32)
    gsin1 = tl.load(base + 2, mask=mask, other=0.0).to(tl.float32)
    gsin2 = tl.load(base + 3, mask=mask, other=0.0).to(tl.float32)
    gcos0 = tl.load(base + 4, mask=mask, other=0.0).to(tl.float32)
    gcos1 = tl.load(base + 5, mask=mask, other=0.0).to(tl.float32)
    gcos2 = tl.load(base + 6, mask=mask, other=0.0).to(tl.float32)

    dangle = (
        gsin0 * libdevice.cos(a0) * f0
        + gsin1 * libdevice.cos(a1) * f1
        + gsin2 * libdevice.cos(a2) * f2
        - gcos0 * libdevice.sin(a0) * f0
        - gcos1 * libdevice.sin(a1) * f1
        - gcos2 * libdevice.sin(a2) * f2
    ) * inv_scale
    denom = tl.sqrt(tl.maximum(1.0 - theta * theta, 1.0e-24))
    dtheta = -dangle / denom
    ddot = dtheta * 0.999999

    gt0 = ddot * sj0
    gt1 = ddot * sj1
    gt2 = ddot * sj2
    gs0 = ddot * ti0
    gs1 = ddot * ti1
    gs2 = ddot * ti2

    tl.atomic_add(grad_unit + tgt * 3 + 0, gt0, mask=mask, sem="relaxed")
    tl.atomic_add(grad_unit + tgt * 3 + 1, gt1, mask=mask, sem="relaxed")
    tl.atomic_add(grad_unit + tgt * 3 + 2, gt2, mask=mask, sem="relaxed")
    tl.atomic_add(grad_unit + src * 3 + 0, gs0, mask=mask, sem="relaxed")
    tl.atomic_add(grad_unit + src * 3 + 1, gs1, mask=mask, sem="relaxed")
    tl.atomic_add(grad_unit + src * 3 + 2, gs2, mask=mask, sem="relaxed")


class _ThreebodyFourierBasis(Function):
    @staticmethod
    def forward(
        ctx,
        unit_vec: torch.Tensor,
        target_index: torch.Tensor,
        source_index: torch.Tensor,
        freqs: torch.Tensor,
        scale: float,
    ):
        if unit_vec.dim() != 2 or unit_vec.shape[1] != 3:
            raise ValueError(f"expected unit_vec [E,3], got {unit_vec.shape}")
        unit_vec = unit_vec.contiguous()
        target_index = target_index.contiguous()
        source_index = source_index.contiguous()
        freqs = freqs.contiguous()
        n = target_index.numel()
        out = torch.empty((n, 7), device=unit_vec.device, dtype=unit_vec.dtype)
        grid = (triton.cdiv(n, _BM),)
        _threebody_fourier_fwd[grid](
            unit_vec,
            target_index,
            source_index,
            freqs,
            out,
            N=n,
            SCALE=float(scale),
            BM=_BM,
            num_warps=4,
        )
        ctx.save_for_backward(unit_vec, target_index, source_index, freqs)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        unit_vec, target_index, source_index, freqs = ctx.saved_tensors
        gout = gout.contiguous()
        n = target_index.numel()
        grad_unit = torch.zeros_like(unit_vec)
        grid = (triton.cdiv(n, _BM),)
        _threebody_fourier_bwd[grid](
            gout,
            unit_vec,
            target_index,
            source_index,
            freqs,
            grad_unit,
            N=n,
            SCALE=ctx.scale,
            BM=_BM,
            num_warps=4,
        )
        return grad_unit, None, None, None, None


def threebody_fourier_basis(
    unit_vec: torch.Tensor,
    target_index: torch.Tensor,
    source_index: torch.Tensor,
    freqs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if unit_vec.is_cuda and target_index.is_cuda and source_index.is_cuda and freqs.is_cuda:
        return _ThreebodyFourierBasis.apply(
            unit_vec, target_index, source_index, freqs, float(scale)
        )
    edge_i = torch.index_select(unit_vec, 0, target_index)
    edge_j = torch.index_select(unit_vec, 0, source_index)
    theta = torch.sum(edge_i * edge_j, dim=1) * _EPS_SCALE
    angle = torch.acos(theta)
    result = angle.new_empty((angle.shape[0], 7))
    result[:, 0] = 2**-0.5
    tmp = torch.outer(angle, freqs)
    result[:, 1:4] = torch.sin(tmp)
    result[:, 4:7] = torch.cos(tmp)
    return result / float(scale)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for n, e in [(1024, 256), (152446, 26174)]:
        unit = torch.nn.functional.normalize(
            torch.randn(e, 3, device=dev), dim=1
        ).requires_grad_(True)
        unit_tri = unit.detach().clone().requires_grad_(True)
        target = torch.randint(0, e, (n,), device=dev)
        source = torch.randint(0, e, (n,), device=dev)
        source = torch.where(source == target, (source + 1) % e, source)
        freqs = torch.tensor([0.833447, 1.5861181, 2.5042067], device=dev)
        scale = 3.141592653589793**0.5
        edge_i = torch.index_select(unit, 0, target)
        edge_j = torch.index_select(unit, 0, source)
        theta = torch.sum(edge_i * edge_j, dim=1) * _EPS_SCALE
        angle = torch.acos(theta)
        ref = angle.new_empty((n, 7))
        ref[:, 0] = 2**-0.5
        tmp = torch.outer(angle, freqs)
        ref[:, 1:4] = torch.sin(tmp)
        ref[:, 4:7] = torch.cos(tmp)
        ref = ref / scale
        tri = threebody_fourier_basis(unit_tri, target, source, freqs, scale)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        print(
            f"N={n} E={e}: "
            f"fwd={(ref - tri).abs().max().item():.2e} "
            f"bwd={(unit.grad - unit_tri.grad).abs().max().item():.2e}"
        )
