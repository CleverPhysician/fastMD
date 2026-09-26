"""Fused gather+cat helpers for graph-captured interaction blocks."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _aligned_gather_cat3_fwd(s0, s1, i1, i2, out, E, D: tl.constexpr, BLK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLK)
    mask = c < D
    b = tl.load(i1 + row)
    d = tl.load(i2 + row)
    v0 = tl.load(s0 + row * D + c, mask=mask, other=0.0)
    v1 = tl.load(s1 + b * D + c, mask=mask, other=0.0)
    v2 = tl.load(s1 + d * D + c, mask=mask, other=0.0)
    o = out + row * (3 * D)
    tl.store(o + c, v0, mask=mask)
    tl.store(o + D + c, v1, mask=mask)
    tl.store(o + 2 * D + c, v2, mask=mask)


@triton.jit
def _aligned_gather_cat3_bwd(s0_grad, s1_grad, i1, i2, gout, E, D: tl.constexpr, BLK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLK)
    mask = c < D
    b = tl.load(i1 + row)
    d = tl.load(i2 + row)
    go = gout + row * (3 * D)
    g0 = tl.load(go + c, mask=mask, other=0.0)
    g1 = tl.load(go + D + c, mask=mask, other=0.0)
    g2 = tl.load(go + 2 * D + c, mask=mask, other=0.0)
    tl.store(s0_grad + row * D + c, g0, mask=mask)
    tl.atomic_add(s1_grad + b * D + c, g1, mask=mask, sem="relaxed")
    tl.atomic_add(s1_grad + d * D + c, g2, mask=mask, sem="relaxed")


class _AlignedGatherCat3(Function):
    @staticmethod
    def forward(ctx, aligned, gathered, index_a, index_b):
        aligned = aligned.contiguous()
        gathered = gathered.contiguous()
        index_a = index_a.contiguous()
        index_b = index_b.contiguous()
        rows, dim = aligned.shape
        block = triton.next_power_of_2(dim)
        out = torch.empty(rows, 3 * dim, device=aligned.device, dtype=aligned.dtype)
        _aligned_gather_cat3_fwd[(rows,)](
            aligned, gathered, index_a, index_b, out, rows, D=dim, BLK=block, num_warps=1
        )
        ctx.save_for_backward(index_a, index_b)
        ctx.shapes = (aligned.shape, gathered.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        index_a, index_b = ctx.saved_tensors
        aligned_shape, gathered_shape = ctx.shapes
        dim = aligned_shape[1]
        block = triton.next_power_of_2(dim)
        grad_aligned = torch.empty(aligned_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_gathered = torch.zeros(gathered_shape, device=grad_out.device, dtype=grad_out.dtype)
        _aligned_gather_cat3_bwd[(aligned_shape[0],)](
            grad_aligned,
            grad_gathered,
            index_a,
            index_b,
            grad_out.contiguous(),
            aligned_shape[0],
            D=dim,
            BLK=block,
            num_warps=1,
        )
        return grad_aligned, grad_gathered, None, None


def aligned_gather_cat3(
    aligned: torch.Tensor,
    gathered: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
) -> torch.Tensor:
    if aligned.is_cuda and gathered.is_cuda:
        return _AlignedGatherCat3.apply(aligned, gathered, index_a, index_b)
    return torch.cat(
        [
            aligned,
            torch.index_select(gathered, 0, index_a),
            torch.index_select(gathered, 0, index_b),
        ],
        dim=1,
    )


@triton.jit
def _aligned_gather_cat4_fwd(
    s0,
    s1,
    i1,
    s2,
    i2,
    i3,
    out,
    E,
    D: tl.constexpr,
    BLK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLK)
    mask = c < D
    a = tl.load(i1 + row)
    b = tl.load(i2 + row)
    d = tl.load(i3 + row)
    v0 = tl.load(s0 + row * D + c, mask=mask, other=0.0)
    v1 = tl.load(s1 + a * D + c, mask=mask, other=0.0)
    v2 = tl.load(s2 + b * D + c, mask=mask, other=0.0)
    v3 = tl.load(s2 + d * D + c, mask=mask, other=0.0)
    o = out + row * (4 * D)
    tl.store(o + c, v0, mask=mask)
    tl.store(o + D + c, v1, mask=mask)
    tl.store(o + 2 * D + c, v2, mask=mask)
    tl.store(o + 3 * D + c, v3, mask=mask)


@triton.jit
def _aligned_gather_cat4_bwd(
    s0_grad,
    s1_grad,
    s2_grad,
    i1,
    i2,
    i3,
    gout,
    E,
    D: tl.constexpr,
    BLK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLK)
    mask = c < D
    a = tl.load(i1 + row)
    b = tl.load(i2 + row)
    d = tl.load(i3 + row)
    go = gout + row * (4 * D)
    g0 = tl.load(go + c, mask=mask, other=0.0)
    g1 = tl.load(go + D + c, mask=mask, other=0.0)
    g2 = tl.load(go + 2 * D + c, mask=mask, other=0.0)
    g3 = tl.load(go + 3 * D + c, mask=mask, other=0.0)
    tl.store(s0_grad + row * D + c, g0, mask=mask)
    tl.atomic_add(s1_grad + a * D + c, g1, mask=mask, sem="relaxed")
    tl.atomic_add(s2_grad + b * D + c, g2, mask=mask, sem="relaxed")
    tl.atomic_add(s2_grad + d * D + c, g3, mask=mask, sem="relaxed")


class _AlignedGatherCat4(Function):
    @staticmethod
    def forward(ctx, aligned, gathered_a, index_a, gathered_b, index_b, index_c):
        aligned = aligned.contiguous()
        gathered_a = gathered_a.contiguous()
        gathered_b = gathered_b.contiguous()
        index_a = index_a.contiguous()
        index_b = index_b.contiguous()
        index_c = index_c.contiguous()
        rows, dim = aligned.shape
        block = triton.next_power_of_2(dim)
        out = torch.empty(rows, 4 * dim, device=aligned.device, dtype=aligned.dtype)
        _aligned_gather_cat4_fwd[(rows,)](
            aligned,
            gathered_a,
            index_a,
            gathered_b,
            index_b,
            index_c,
            out,
            rows,
            D=dim,
            BLK=block,
            num_warps=1,
        )
        ctx.save_for_backward(index_a, index_b, index_c)
        ctx.shapes = (aligned.shape, gathered_a.shape, gathered_b.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        index_a, index_b, index_c = ctx.saved_tensors
        aligned_shape, gathered_a_shape, gathered_b_shape = ctx.shapes
        dim = aligned_shape[1]
        block = triton.next_power_of_2(dim)
        grad_aligned = torch.empty(aligned_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_gathered_a = torch.zeros(gathered_a_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_gathered_b = torch.zeros(gathered_b_shape, device=grad_out.device, dtype=grad_out.dtype)
        _aligned_gather_cat4_bwd[(aligned_shape[0],)](
            grad_aligned,
            grad_gathered_a,
            grad_gathered_b,
            index_a,
            index_b,
            index_c,
            grad_out.contiguous(),
            aligned_shape[0],
            D=dim,
            BLK=block,
            num_warps=1,
        )
        return grad_aligned, grad_gathered_a, None, grad_gathered_b, None, None


def aligned_gather_cat4(
    aligned: torch.Tensor,
    gathered_a: torch.Tensor,
    index_a: torch.Tensor,
    gathered_b: torch.Tensor,
    index_b: torch.Tensor,
    index_c: torch.Tensor,
) -> torch.Tensor:
    if aligned.is_cuda and gathered_a.is_cuda and gathered_b.is_cuda:
        return _AlignedGatherCat4.apply(aligned, gathered_a, index_a, gathered_b, index_b, index_c)
    return torch.cat(
        [
            aligned,
            torch.index_select(gathered_a, 0, index_a),
            torch.index_select(gathered_b, 0, index_b),
            torch.index_select(gathered_b, 0, index_c),
        ],
        dim=1,
    )
