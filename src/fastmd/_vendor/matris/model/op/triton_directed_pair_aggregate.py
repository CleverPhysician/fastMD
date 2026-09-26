"""Triton specialization for directed-edge -> undirected-edge averaging.

The atom graph stores exactly two directed edges for each undirected edge.  The
generic path uses ``index_add_(directed2undirected) / bincount``.  This operator
uses a precomputed ``pair_index[U, 2]`` and performs:

    out[u, d] = 0.5 * (data[pair_index[u, 0], d] + data[pair_index[u, 1], d])
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


_PAIR_AGG_BM = _env_int("MATRIS_DIRECTED_PAIR_AGG_BM", 8)
_PAIR_AGG_BD = _env_int("MATRIS_DIRECTED_PAIR_AGG_BD", 128)


@triton.jit
def _directed_pair_average_fwd(
    data,
    pair_index,
    out,
    U,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < U) & (cols[None, :] < D)
    idx0 = tl.load(pair_index + rows * 2, mask=rows < U, other=0)
    idx1 = tl.load(pair_index + rows * 2 + 1, mask=rows < U, other=0)
    v0 = tl.load(data + idx0[:, None] * D + cols[None, :], mask=mask, other=0.0)
    v1 = tl.load(data + idx1[:, None] * D + cols[None, :], mask=mask, other=0.0)
    tl.store(out + rows[:, None] * D + cols[None, :], (v0 + v1) * 0.5, mask=mask)


@triton.jit
def _directed_pair_average_bwd(
    gout,
    pair_index,
    grad_data,
    U,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < U) & (cols[None, :] < D)
    idx0 = tl.load(pair_index + rows * 2, mask=rows < U, other=0)
    idx1 = tl.load(pair_index + rows * 2 + 1, mask=rows < U, other=0)
    grad = tl.load(gout + rows[:, None] * D + cols[None, :], mask=mask, other=0.0)
    grad *= 0.5
    tl.store(grad_data + idx0[:, None] * D + cols[None, :], grad, mask=mask)
    tl.store(grad_data + idx1[:, None] * D + cols[None, :], grad, mask=mask)


@triton.jit
def _undirected_pair_expand_fwd(
    data,
    pair_index,
    out,
    U,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < U) & (cols[None, :] < D)
    idx0 = tl.load(pair_index + rows * 2, mask=rows < U, other=0)
    idx1 = tl.load(pair_index + rows * 2 + 1, mask=rows < U, other=0)
    vals = tl.load(data + rows[:, None] * D + cols[None, :], mask=mask, other=0.0)
    tl.store(out + idx0[:, None] * D + cols[None, :], vals, mask=mask)
    tl.store(out + idx1[:, None] * D + cols[None, :], vals, mask=mask)


@triton.jit
def _undirected_pair_expand_bwd(
    gout,
    pair_index,
    grad_data,
    U,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < U) & (cols[None, :] < D)
    idx0 = tl.load(pair_index + rows * 2, mask=rows < U, other=0)
    idx1 = tl.load(pair_index + rows * 2 + 1, mask=rows < U, other=0)
    g0 = tl.load(gout + idx0[:, None] * D + cols[None, :], mask=mask, other=0.0)
    g1 = tl.load(gout + idx1[:, None] * D + cols[None, :], mask=mask, other=0.0)
    tl.store(grad_data + rows[:, None] * D + cols[None, :], g0 + g1, mask=mask)


class _DirectedPairAverage(Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor, pair_index: torch.Tensor):
        if data.dim() != 2 or pair_index.dim() != 2 or pair_index.shape[1] != 2:
            raise ValueError(
                f"expected data [E,D] and pair_index [U,2], got {data.shape}, {pair_index.shape}"
            )
        data = data.contiguous() if data.stride(1) != 1 else data
        pair_index = pair_index.contiguous()
        u, d = pair_index.shape[0], data.shape[1]
        out = torch.empty((u, d), device=data.device, dtype=data.dtype)
        _directed_pair_average_fwd[
            (triton.cdiv(u, _PAIR_AGG_BM), triton.cdiv(d, _PAIR_AGG_BD))
        ](
            data,
            pair_index,
            out,
            U=u,
            D=d,
            BM=_PAIR_AGG_BM,
            BD=triton.next_power_of_2(min(_PAIR_AGG_BD, d)),
            num_warps=4,
        )
        ctx.save_for_backward(pair_index)
        ctx.data_shape = data.shape
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (pair_index,) = ctx.saved_tensors
        e, d = ctx.data_shape
        u = pair_index.shape[0]
        gout = gout.contiguous()
        grad_data = torch.empty((e, d), device=gout.device, dtype=gout.dtype)
        _directed_pair_average_bwd[
            (triton.cdiv(u, _PAIR_AGG_BM), triton.cdiv(d, _PAIR_AGG_BD))
        ](
            gout,
            pair_index,
            grad_data,
            U=u,
            D=d,
            BM=_PAIR_AGG_BM,
            BD=triton.next_power_of_2(min(_PAIR_AGG_BD, d)),
            num_warps=4,
        )
        return grad_data, None


class _UndirectedPairExpand(Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor, pair_index: torch.Tensor, num_directed: int):
        if data.dim() != 2 or pair_index.dim() != 2 or pair_index.shape[1] != 2:
            raise ValueError(
                f"expected data [U,D] and pair_index [U,2], got {data.shape}, {pair_index.shape}"
            )
        data = data.contiguous() if data.stride(1) != 1 else data
        pair_index = pair_index.contiguous()
        u, d = data.shape
        out = torch.empty((int(num_directed), d), device=data.device, dtype=data.dtype)
        _undirected_pair_expand_fwd[
            (triton.cdiv(u, _PAIR_AGG_BM), triton.cdiv(d, _PAIR_AGG_BD))
        ](
            data,
            pair_index,
            out,
            U=u,
            D=d,
            BM=_PAIR_AGG_BM,
            BD=triton.next_power_of_2(min(_PAIR_AGG_BD, d)),
            num_warps=4,
        )
        ctx.save_for_backward(pair_index)
        ctx.data_shape = data.shape
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (pair_index,) = ctx.saved_tensors
        u, d = ctx.data_shape
        gout = gout.contiguous()
        grad_data = torch.empty((u, d), device=gout.device, dtype=gout.dtype)
        _undirected_pair_expand_bwd[
            (triton.cdiv(u, _PAIR_AGG_BM), triton.cdiv(d, _PAIR_AGG_BD))
        ](
            gout,
            pair_index,
            grad_data,
            U=u,
            D=d,
            BM=_PAIR_AGG_BM,
            BD=triton.next_power_of_2(min(_PAIR_AGG_BD, d)),
            num_warps=4,
        )
        return grad_data, None, None


def directed_pair_average(
    data: torch.Tensor,
    pair_index: torch.Tensor,
) -> torch.Tensor:
    if data.is_cuda and pair_index.is_cuda:
        return _DirectedPairAverage.apply(data, pair_index)
    return (data[pair_index[:, 0]] + data[pair_index[:, 1]]) * 0.5


def undirected_pair_expand(
    data: torch.Tensor,
    pair_index: torch.Tensor,
    num_directed: int,
) -> torch.Tensor:
    if data.is_cuda and pair_index.is_cuda:
        return _UndirectedPairExpand.apply(data, pair_index, int(num_directed))
    out = data.new_empty((int(num_directed), data.shape[1]))
    out[pair_index[:, 0]] = data
    out[pair_index[:, 1]] = data
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for e, d in [(1024, 128), (26174, 128), (27648, 128)]:
        u = e // 2
        seg = torch.arange(u, device=dev, dtype=torch.long).repeat_interleave(2)
        perm = torch.randperm(e, device=dev)
        seg = seg[perm]
        pair = torch.argsort(seg).reshape(u, 2).contiguous()
        x = torch.randn(e, d, device=dev, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        ref = x_ref.new_zeros((u, d)).index_add_(0, seg, x_ref) * 0.5
        tri = directed_pair_average(x, pair)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        torch.cuda.synchronize()
        print(
            f"E={e} D={d}: fwd={float((ref - tri).abs().max().detach()):.2e} "
            f"bwd={float((x_ref.grad - x.grad).abs().max().detach()):.2e}"
        )

        x = torch.randn(u, d, device=dev, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        ref = torch.index_select(x_ref, 0, seg)
        tri = undirected_pair_expand(x, pair, e)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        torch.cuda.synchronize()
        print(
            f"E={e} D={d} expand: fwd={float((ref - tri).abs().max().detach()):.2e} "
            f"bwd={float((x_ref.grad - x.grad).abs().max().detach()):.2e}"
        )
