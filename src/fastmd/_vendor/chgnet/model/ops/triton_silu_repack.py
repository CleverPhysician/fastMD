"""Fused SiLU activation and branch repacking for frozen CHGNet GatedMLPs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _silu_repack_fwd(
    packed,
    out,
    rows,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)[None, :]
    mask = (row < rows) & (cols < H)

    packed_base = row * (2 * H)
    out_base = row * H
    core = tl.load(packed + packed_base + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(packed + packed_base + H + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    core = core * tl.sigmoid(core)
    gate = gate * tl.sigmoid(gate)
    tl.store(out + out_base + cols, core, mask=mask)
    tl.store(out + rows * H + out_base + cols, gate, mask=mask)


@triton.jit
def _silu_repack_bwd(
    grad_out,
    packed,
    grad_packed,
    rows,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = 2 * H
    mask = offsets < rows * width
    row = offsets // width
    feature = offsets - row * width
    is_gate = feature >= H
    branch_feature = tl.where(is_gate, feature - H, feature)
    grad_offsets = row * H + branch_feature + tl.where(
        is_gate, rows * H, 0
    )

    values = tl.load(packed + offsets, mask=mask, other=0.0).to(tl.float32)
    gradients = tl.load(grad_out + grad_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sigmoid = tl.sigmoid(values)
    derivative = sigmoid * (1.0 + values * (1.0 - sigmoid))
    tl.store(grad_packed + offsets, gradients * derivative, mask=mask)


def _launch_config(half_width: int) -> tuple[int, int, int]:
    """Return bounded feature/row blocks and a matching warp count."""
    block = min(triton.next_power_of_2(half_width), 256)
    # Amortize program scheduling for the common narrow (H=64) CHGNet
    # projection without allowing a tile to grow beyond 512 values per branch.
    row_block = max(1, 512 // block)
    warps = 4 if block * row_block <= 512 else 8
    return block, row_block, warps


class _SiLURepack(Function):
    @staticmethod
    def forward(ctx, packed):
        rows, width = packed.shape
        half_width = width // 2
        out = torch.empty(
            (2, rows, half_width),
            dtype=packed.dtype,
            device=packed.device,
        )
        block, row_block, warps = _launch_config(half_width)
        grid = (triton.cdiv(rows, row_block), triton.cdiv(half_width, block))
        _silu_repack_fwd[grid](
            packed,
            out,
            rows,
            H=half_width,
            BLOCK=block,
            ROW_BLOCK=row_block,
            num_warps=warps,
        )
        ctx.save_for_backward(packed)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (packed,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_packed = torch.empty_like(
            packed,
            memory_format=torch.contiguous_format,
        )
        rows, width = packed.shape
        half_width = width // 2
        block = 256
        grid = (triton.cdiv(rows * width, block),)
        _silu_repack_bwd[grid](
            grad_out,
            packed,
            grad_packed,
            rows,
            H=half_width,
            BLOCK=block,
            num_warps=4,
        )
        return grad_packed


def silu_repack(packed: torch.Tensor) -> torch.Tensor:
    """Apply SiLU and repack ``[N, 2H]`` into contiguous ``[2, N, H]``.

    CUDA float32 contiguous matrices use a graph-safe Triton implementation.
    Other floating-point inputs use native PyTorch operations while preserving
    the same output layout and autograd semantics.
    """
    if packed.ndim != 2:
        raise ValueError("packed must be a rank-2 matrix with shape (N, 2H)")
    if not packed.is_floating_point():
        raise TypeError("packed must use a floating-point dtype")
    rows, width = packed.shape
    if width % 2 != 0:
        raise ValueError("packed feature width must be even")
    half_width = width // 2
    use_triton = (
        packed.is_cuda
        and packed.dtype == torch.float32
        and rows > 0
        and half_width > 0
        and packed.is_contiguous()
        and packed.stride(1) == 1
    )
    if use_triton:
        return _SiLURepack.apply(packed)
    activated = F.silu(packed)
    return activated.reshape(rows, 2, half_width).permute(1, 0, 2).contiguous()


__all__ = ["silu_repack"]
