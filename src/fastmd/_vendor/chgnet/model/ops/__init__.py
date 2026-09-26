"""Optional fused CUDA operators used by CHGNet inference backends."""

from __future__ import annotations

from fastmd._vendor.chgnet.model.ops.triton_gather_cat import (
    gather_cat_atom_messages,
    gather_cat_bond_messages,
)
from fastmd._vendor.chgnet.model.ops.triton_gated_mlp import layer_norm_silu_gate
from fastmd._vendor.chgnet.model.ops.triton_indexed_weighted_segment_sum import (
    indexed_triple_weighted_segment_sum,
)
from fastmd._vendor.chgnet.model.ops.triton_silu_repack import silu_repack
from fastmd._vendor.chgnet.model.ops.triton_weighted_segment_sum import (
    triple_weighted_segment_sum,
    weighted_segment_sum,
)

__all__ = [
    "gather_cat_atom_messages",
    "gather_cat_bond_messages",
    "indexed_triple_weighted_segment_sum",
    "layer_norm_silu_gate",
    "silu_repack",
    "triple_weighted_segment_sum",
    "weighted_segment_sum",
]
