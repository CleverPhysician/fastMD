"""Fused gather and concatenation kernels for CHGNet message construction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function


def _validate_feature_matrix(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 feature matrix")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def _validate_index_matrix(
    name: str,
    value: torch.Tensor,
    *,
    columns: int,
) -> None:
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64 indices")


def _require_same_device(*values: torch.Tensor) -> None:
    devices = {value.device for value in values}
    if len(devices) != 1:
        raise ValueError("fused message inputs must be on the same device")


@triton.jit
def _atom_message_fwd(
    atom,
    bond,
    center,
    neighbor,
    bond_index,
    out,
    DA: tl.constexpr,
    DB: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    CENTER_STRIDE: tl.constexpr,
    NEIGHBOR_STRIDE: tl.constexpr,
    BOND_INDEX_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    ca = tl.arange(0, BA)
    cb = tl.arange(0, BB)
    ma = ca < DA
    mb = cb < DB
    i = tl.load(center + row * CENTER_STRIDE)
    j = tl.load(neighbor + row * NEIGHBOR_STRIDE)
    b = tl.load(bond_index + row * BOND_INDEX_STRIDE)
    va = tl.load(atom + i * DA + ca, mask=ma, other=0.0)
    vb = tl.load(bond + b * DB + cb, mask=mb, other=0.0)
    vn = tl.load(atom + j * DA + ca, mask=ma, other=0.0)
    base = out + row * (2 * DA + DB)
    tl.store(base + ca, va, mask=ma)
    tl.store(base + DA + cb, vb, mask=mb)
    tl.store(base + DA + DB + ca, vn, mask=ma)


@triton.jit
def _atom_message_bwd(
    grad_out,
    center,
    neighbor,
    bond_index,
    grad_atom,
    grad_bond,
    DA: tl.constexpr,
    DB: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    CENTER_STRIDE: tl.constexpr,
    NEIGHBOR_STRIDE: tl.constexpr,
    BOND_INDEX_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    ca = tl.arange(0, BA)
    cb = tl.arange(0, BB)
    ma = ca < DA
    mb = cb < DB
    i = tl.load(center + row * CENTER_STRIDE)
    j = tl.load(neighbor + row * NEIGHBOR_STRIDE)
    b = tl.load(bond_index + row * BOND_INDEX_STRIDE)
    base = grad_out + row * (2 * DA + DB)
    gi = tl.load(base + ca, mask=ma, other=0.0)
    gb = tl.load(base + DA + cb, mask=mb, other=0.0)
    gj = tl.load(base + DA + DB + ca, mask=ma, other=0.0)
    tl.atomic_add(grad_atom + i * DA + ca, gi, mask=ma, sem="relaxed")
    tl.atomic_add(grad_bond + b * DB + cb, gb, mask=mb, sem="relaxed")
    tl.atomic_add(grad_atom + j * DA + ca, gj, mask=ma, sem="relaxed")


class _GatherCatAtomMessages(Function):
    @staticmethod
    def forward(ctx, atom, bond, center, neighbor, bond_index):
        atom = atom.contiguous()
        bond = bond.contiguous()
        rows = center.shape[0]
        da = atom.shape[1]
        db = bond.shape[1]
        out = atom.new_empty((rows, 2 * da + db))
        _atom_message_fwd[(rows,)](
            atom,
            bond,
            center,
            neighbor,
            bond_index,
            out,
            DA=da,
            DB=db,
            BA=triton.next_power_of_2(da),
            BB=triton.next_power_of_2(db),
            CENTER_STRIDE=center.stride(0),
            NEIGHBOR_STRIDE=neighbor.stride(0),
            BOND_INDEX_STRIDE=bond_index.stride(0),
            num_warps=4,
        )
        ctx.save_for_backward(center, neighbor, bond_index)
        ctx.input_shapes = (atom.shape, bond.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        center, neighbor, bond_index = ctx.saved_tensors
        atom_shape, bond_shape = ctx.input_shapes
        grad_out = grad_out.contiguous()
        grad_atom = grad_out.new_zeros(atom_shape)
        grad_bond = grad_out.new_zeros(bond_shape)
        _atom_message_bwd[(center.shape[0],)](
            grad_out,
            center,
            neighbor,
            bond_index,
            grad_atom,
            grad_bond,
            DA=atom_shape[1],
            DB=bond_shape[1],
            BA=triton.next_power_of_2(atom_shape[1]),
            BB=triton.next_power_of_2(bond_shape[1]),
            CENTER_STRIDE=center.stride(0),
            NEIGHBOR_STRIDE=neighbor.stride(0),
            BOND_INDEX_STRIDE=bond_index.stride(0),
            num_warps=4,
        )
        return grad_atom, grad_bond, None, None, None


def gather_cat_atom_messages(
    atom: torch.Tensor,
    bond: torch.Tensor,
    atom_graph: torch.Tensor,
    directed2undirected: torch.Tensor,
) -> torch.Tensor:
    """Return ``[atom[center], bond[edge], atom[neighbor]]`` per directed edge."""
    _validate_feature_matrix("atom", atom)
    _validate_feature_matrix("bond", bond)
    _validate_index_matrix("atom_graph", atom_graph, columns=2)
    if directed2undirected.ndim != 1:
        raise ValueError("directed2undirected must be one-dimensional")
    if directed2undirected.dtype not in (torch.int32, torch.int64):
        raise TypeError("directed2undirected must use int32 or int64 indices")
    if atom_graph.shape[0] != directed2undirected.shape[0]:
        raise ValueError("atom_graph and directed2undirected row counts must match")
    if atom.dtype != bond.dtype:
        raise TypeError("atom and bond features must have the same dtype")
    _require_same_device(atom, bond, atom_graph, directed2undirected)

    center = atom_graph[:, 0]
    neighbor = atom_graph[:, 1]
    if atom.is_cuda and center.numel() > 0 and atom.shape[1] > 0 and bond.shape[1] > 0:
        return _GatherCatAtomMessages.apply(
            atom, bond, center, neighbor, directed2undirected
        )
    return torch.cat(
        [
            torch.index_select(atom, 0, center),
            torch.index_select(bond, 0, directed2undirected),
            torch.index_select(atom, 0, neighbor),
        ],
        dim=1,
    )


@triton.jit
def _bond_message_fwd(
    bond,
    angle,
    atom,
    bond_i,
    bond_j,
    center,
    out,
    DB: tl.constexpr,
    DG: tl.constexpr,
    DA: tl.constexpr,
    BB: tl.constexpr,
    BG: tl.constexpr,
    BA: tl.constexpr,
    BOND_I_STRIDE: tl.constexpr,
    BOND_J_STRIDE: tl.constexpr,
    CENTER_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.arange(0, BB)
    cg = tl.arange(0, BG)
    ca = tl.arange(0, BA)
    mb = cb < DB
    mg = cg < DG
    ma = ca < DA
    i = tl.load(bond_i + row * BOND_I_STRIDE)
    j = tl.load(bond_j + row * BOND_J_STRIDE)
    a = tl.load(center + row * CENTER_STRIDE)
    vi = tl.load(bond + i * DB + cb, mask=mb, other=0.0)
    vj = tl.load(bond + j * DB + cb, mask=mb, other=0.0)
    vg = tl.load(angle + row * DG + cg, mask=mg, other=0.0)
    va = tl.load(atom + a * DA + ca, mask=ma, other=0.0)
    base = out + row * (2 * DB + DG + DA)
    tl.store(base + cb, vi, mask=mb)
    tl.store(base + DB + cb, vj, mask=mb)
    tl.store(base + 2 * DB + cg, vg, mask=mg)
    tl.store(base + 2 * DB + DG + ca, va, mask=ma)


@triton.jit
def _bond_message_bwd(
    grad_out,
    bond_i,
    bond_j,
    center,
    grad_bond,
    grad_angle,
    grad_atom,
    DB: tl.constexpr,
    DG: tl.constexpr,
    DA: tl.constexpr,
    BB: tl.constexpr,
    BG: tl.constexpr,
    BA: tl.constexpr,
    BOND_I_STRIDE: tl.constexpr,
    BOND_J_STRIDE: tl.constexpr,
    CENTER_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.arange(0, BB)
    cg = tl.arange(0, BG)
    ca = tl.arange(0, BA)
    mb = cb < DB
    mg = cg < DG
    ma = ca < DA
    i = tl.load(bond_i + row * BOND_I_STRIDE)
    j = tl.load(bond_j + row * BOND_J_STRIDE)
    a = tl.load(center + row * CENTER_STRIDE)
    base = grad_out + row * (2 * DB + DG + DA)
    gi = tl.load(base + cb, mask=mb, other=0.0)
    gj = tl.load(base + DB + cb, mask=mb, other=0.0)
    gg = tl.load(base + 2 * DB + cg, mask=mg, other=0.0)
    ga = tl.load(base + 2 * DB + DG + ca, mask=ma, other=0.0)
    tl.atomic_add(grad_bond + i * DB + cb, gi, mask=mb, sem="relaxed")
    tl.atomic_add(grad_bond + j * DB + cb, gj, mask=mb, sem="relaxed")
    tl.store(grad_angle + row * DG + cg, gg, mask=mg)
    tl.atomic_add(grad_atom + a * DA + ca, ga, mask=ma, sem="relaxed")


class _GatherCatBondMessages(Function):
    @staticmethod
    def forward(ctx, atom, bond, angle, center, bond_i, bond_j):
        atom = atom.contiguous()
        bond = bond.contiguous()
        angle = angle.contiguous()
        rows = center.shape[0]
        da, db, dg = atom.shape[1], bond.shape[1], angle.shape[1]
        out = atom.new_empty((rows, 2 * db + dg + da))
        _bond_message_fwd[(rows,)](
            bond,
            angle,
            atom,
            bond_i,
            bond_j,
            center,
            out,
            DB=db,
            DG=dg,
            DA=da,
            BB=triton.next_power_of_2(db),
            BG=triton.next_power_of_2(dg),
            BA=triton.next_power_of_2(da),
            BOND_I_STRIDE=bond_i.stride(0),
            BOND_J_STRIDE=bond_j.stride(0),
            CENTER_STRIDE=center.stride(0),
            num_warps=4,
        )
        ctx.save_for_backward(center, bond_i, bond_j)
        ctx.input_shapes = (atom.shape, bond.shape, angle.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        center, bond_i, bond_j = ctx.saved_tensors
        atom_shape, bond_shape, angle_shape = ctx.input_shapes
        grad_out = grad_out.contiguous()
        grad_atom = grad_out.new_zeros(atom_shape)
        grad_bond = grad_out.new_zeros(bond_shape)
        grad_angle = grad_out.new_empty(angle_shape)
        _bond_message_bwd[(center.shape[0],)](
            grad_out,
            bond_i,
            bond_j,
            center,
            grad_bond,
            grad_angle,
            grad_atom,
            DB=bond_shape[1],
            DG=angle_shape[1],
            DA=atom_shape[1],
            BB=triton.next_power_of_2(bond_shape[1]),
            BG=triton.next_power_of_2(angle_shape[1]),
            BA=triton.next_power_of_2(atom_shape[1]),
            BOND_I_STRIDE=bond_i.stride(0),
            BOND_J_STRIDE=bond_j.stride(0),
            CENTER_STRIDE=center.stride(0),
            num_warps=4,
        )
        return grad_atom, grad_bond, grad_angle, None, None, None


def gather_cat_bond_messages(
    atom: torch.Tensor,
    bond: torch.Tensor,
    angle: torch.Tensor,
    bond_graph: torch.Tensor,
) -> torch.Tensor:
    """Return ``[bond[i], bond[j], angle, atom[center]]`` per triplet."""
    _validate_feature_matrix("atom", atom)
    _validate_feature_matrix("bond", bond)
    _validate_feature_matrix("angle", angle)
    _validate_index_matrix("bond_graph", bond_graph, columns=3)
    if bond_graph.shape[0] != angle.shape[0]:
        raise ValueError("bond_graph and angle row counts must match")
    if not (atom.dtype == bond.dtype == angle.dtype):
        raise TypeError("atom, bond, and angle features must have the same dtype")
    _require_same_device(atom, bond, angle, bond_graph)

    center = bond_graph[:, 0]
    bond_i = bond_graph[:, 1]
    bond_j = bond_graph[:, 2]
    if (
        atom.is_cuda
        and center.numel() > 0
        and atom.shape[1] > 0
        and bond.shape[1] > 0
        and angle.shape[1] > 0
    ):
        return _GatherCatBondMessages.apply(atom, bond, angle, center, bond_i, bond_j)
    return torch.cat(
        [
            torch.index_select(bond, 0, bond_i),
            torch.index_select(bond, 0, bond_j),
            angle,
            torch.index_select(atom, 0, center),
        ],
        dim=1,
    )
