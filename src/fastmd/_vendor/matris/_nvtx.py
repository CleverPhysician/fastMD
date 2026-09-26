from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range when CUDA profiling is available."""
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
