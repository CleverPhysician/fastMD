"""GPU compaction of active line-graph rows into a fixed capacity."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_active_triplets(
    line_source_pointer,
    line_destination_pointer,
    row_mask_pointer,
    prefix_pointer,
    compact_source_pointer,
    compact_destination_pointer,
    rows,
    capacity,
    stripe_size,
    interleave: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = row < rows
    active = valid & (
        tl.load(row_mask_pointer + row, mask=valid, other=0.0) != 0.0
    )
    active_rank = tl.load(prefix_pointer + row, mask=active, other=0) - 1
    compact_row = (
        (active_rank % interleave) * stripe_size
        + active_rank // interleave
    )
    write = active & (active_rank < capacity)
    source = tl.load(line_source_pointer + row, mask=write, other=0)
    destination = tl.load(
        line_destination_pointer + row,
        mask=write,
        other=0,
    )
    tl.store(compact_source_pointer + compact_row, source, mask=write)
    tl.store(
        compact_destination_pointer + compact_row,
        destination,
        mask=write,
    )


def compact_active_triplets(
    line_source: torch.Tensor,
    line_destination: torch.Tensor,
    row_mask: torch.Tensor,
    compact_source: torch.Tensor,
    compact_destination: torch.Tensor,
    interleave: int = 128,
) -> torch.Tensor:
    """Pack active triplets and return their untruncated device count."""
    active = row_mask.reshape(-1).to(dtype=torch.int64)
    prefix = torch.cumsum(active, dim=0)
    rows = active.numel()
    if interleave <= 0:
        raise ValueError("interleave must be positive")
    if compact_source.numel() % interleave:
        raise ValueError("compact capacity must be divisible by interleave")
    block_size = 256
    _compact_active_triplets[(triton.cdiv(rows, block_size),)](
        line_source,
        line_destination,
        row_mask,
        prefix,
        compact_source,
        compact_destination,
        rows,
        compact_source.numel(),
        compact_source.numel() // interleave,
        interleave=interleave,
        block_size=block_size,
        num_warps=4,
    )
    return prefix[-1]
