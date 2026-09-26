"""CUDA radius-graph construction using the NVIDIA neighbor-list operator.

This module ports the GPU graph builder introduced in MatRIS commit
50ad204a83c9987b2a0da03a694bc26d51aba4b0.  ALIGNN uses DGL graphs rather
than MatRIS ``RadiusGraph`` objects, so this module returns the directed edge
indices, periodic images, and displacement vectors consumed by
``Graph.atom_dgl_multigraph``.

The CUDA dependencies are optional.  Importing ALIGNN continues to work when
``nvalchemiops`` is not installed; callers can use ``gpu_graph_available`` to
select the CUDA path and otherwise retain the existing CPU implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import dgl
import numpy as np
import torch
from torch import Tensor

try:
    from nvalchemiops.neighborlist.neighbor_utils import (
        estimate_max_neighbors,
    )
    from nvalchemiops.neighborlist.neighborlist import (
        neighbor_list as _nvidia_neighbor_list,
    )

    op_available = True
except ImportError:
    estimate_max_neighbors = None
    _nvidia_neighbor_list = None
    op_available = False


def gpu_graph_available(device: torch.device | str | None = None) -> bool:
    """Return whether the requested device can use the CUDA graph builder."""
    if device is None or not op_available or not torch.cuda.is_available():
        return False
    return torch.device(device).type == "cuda"


def _cuda_device(device: torch.device | str) -> torch.device:
    """Normalize and activate a CUDA device."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"GPU graph construction requires CUDA, got {device}")
    if device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(device)
    return device


def _atoms_arrays(atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract cell, Cartesian positions, and PBC from ASE or JARVIS atoms."""
    if hasattr(atoms, "get_positions"):
        cell = np.asarray(atoms.get_cell().array, dtype=np.float32)
        positions = np.asarray(atoms.get_positions(), dtype=np.float32)
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    else:
        cell = np.asarray(atoms.lattice_mat, dtype=np.float32)
        positions = np.asarray(atoms.cart_coords, dtype=np.float32)
        # JARVIS Atoms represents periodic structures and does not retain the
        # per-axis ASE PBC flags used by the source MatRIS implementation.
        pbc = np.ones(3, dtype=bool)
    return cell, positions, pbc


def _extract_atoms(atoms_list: Sequence, device: torch.device) -> dict:
    """Pack one or more structures into a batched CUDA payload."""
    positions = []
    cells = []
    pbc = []
    batch_idx = []
    atom_offsets = [0]

    for graph_idx, atoms in enumerate(atoms_list):
        cell_array, position_array, pbc_array = _atoms_arrays(atoms)
        cell = torch.as_tensor(cell_array, device=device)
        pos = torch.as_tensor(position_array, device=device)
        frac = (pos @ torch.linalg.inv(cell)) % 1.0
        wrapped_positions = frac @ cell
        n_atoms = len(position_array)

        positions.append(wrapped_positions)
        cells.append(cell)
        pbc.append(torch.as_tensor(pbc_array, device=device))
        batch_idx.append(
            torch.full(
                (n_atoms,),
                graph_idx,
                dtype=torch.int32,
                device=device,
            )
        )
        atom_offsets.append(atom_offsets[-1] + n_atoms)

    return {
        "positions": torch.cat(positions),
        "cells": torch.stack(cells),
        "pbc": torch.stack(pbc).bool(),
        "batch_idx": torch.cat(batch_idx),
        "atom_offsets": atom_offsets,
    }


@torch.no_grad()
def neighbor_list_nvidia(
    positions: Tensor,
    cell: Tensor,
    cutoff: float,
    batch_idx: Tensor,
    pbc: Tensor,
    device: torch.device | str = "cuda",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build and deterministically sort a periodic CUDA neighbor list."""
    if not op_available:
        raise ImportError(
            "GPU graph construction requires nvalchemi-toolkit-ops."
        )

    device = _cuda_device(device)
    positions = positions.to(device)
    cell = cell.to(device)
    batch_idx = batch_idx.to(device=device, dtype=torch.int32)
    pbc = pbc.to(device=device, dtype=torch.bool)

    n_atoms = positions.shape[0]
    buffer_max = estimate_max_neighbors(cutoff=cutoff, safety_factor=2.0)
    neighbor_matrix = torch.full(
        (n_atoms, buffer_max),
        n_atoms,
        dtype=torch.int32,
        device=device,
    )
    neighbor_shifts = torch.zeros(
        (n_atoms, buffer_max, 3),
        dtype=torch.int32,
        device=device,
    )
    num_neighbors = torch.zeros(n_atoms, dtype=torch.int32, device=device)

    _nvidia_neighbor_list(
        positions=positions,
        cutoff=cutoff + 1e-6,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        method="batch_cell_list",
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_shifts,
        num_neighbors=num_neighbors,
        half_fill=False,
    )

    atom_idx = torch.arange(n_atoms, device=device).unsqueeze(1)
    neighbor_slot = torch.arange(buffer_max, device=device).unsqueeze(0)
    valid = neighbor_slot < num_neighbors.unsqueeze(1)
    center = atom_idx.expand(-1, buffer_max)[valid].long()
    neighbor = neighbor_matrix[valid].long()
    images = neighbor_shifts[valid]

    if center.numel() == 0:
        empty_index = torch.empty(0, dtype=torch.long, device=device)
        empty_image = torch.empty((0, 3), dtype=torch.int32, device=device)
        empty_distance = torch.empty(0, dtype=positions.dtype, device=device)
        return empty_index, empty_index, empty_image, empty_distance

    edge_batch = batch_idx[center].long()
    displacement = positions[neighbor] - positions[center]
    displacement = displacement + torch.einsum(
        "ei,eij->ej", images.float(), cell[edge_batch]
    )
    distance = displacement.norm(dim=1)
    mask = distance < cutoff
    center = center[mask]
    neighbor = neighbor[mask]
    images = images[mask]
    distance = distance[mask]

    if center.numel() == 0:
        return center, neighbor, images, distance

    image_key = images.long()
    max_image = max(int(image_key.abs().max().item()), 10)
    radix = 2 * max_image + 1
    order = (
        batch_idx[center].long() * n_atoms * n_atoms * radix**3
        + center * n_atoms * radix**3
        + neighbor * radix**3
        + (image_key[:, 0] + max_image) * radix * radix
        + (image_key[:, 1] + max_image) * radix
        + image_key[:, 2]
        + max_image
    ).argsort()
    return center[order], neighbor[order], images[order], distance[order]


@torch.no_grad()
def radius_graph_gpu(
    atoms,
    cutoff: float = 5.0,
    cutoff_extra: float = 0.5,
    device: torch.device | str = "cuda",
) -> tuple[Tensor, Tensor, Tensor, Tensor] | list[
    tuple[Tensor, Tensor, Tensor, Tensor]
]:
    """Return CUDA radius-graph edges for one structure or a structure batch.

    Returns ``(source, destination, displacement, periodic_image)``.  A list of
    those tuples is returned when ``atoms`` is a sequence of structures.
    """
    device = _cuda_device(device)
    batched = not (
        hasattr(atoms, "get_positions") or hasattr(atoms, "cart_coords")
    )
    atoms_list = list(atoms) if batched else [atoms]
    if not atoms_list:
        return []

    if cutoff_extra <= 0:
        raise ValueError("cutoff_extra must be positive")

    payload = _extract_atoms(atoms_list, device)
    n_atoms = int(payload["positions"].shape[0])
    graph_cutoff = cutoff
    while True:
        center, neighbor, images, _ = neighbor_list_nvidia(
            payload["positions"],
            payload["cells"],
            graph_cutoff,
            payload["batch_idx"],
            payload["pbc"],
            device=device,
        )
        counts = torch.bincount(center, minlength=n_atoms)
        if torch.all(counts > 0):
            break
        # Match ALIGNN's existing CPU radius_graph behavior: enlarge the
        # radius until every atom participates in at least one bond.
        graph_cutoff += cutoff_extra

    edge_batch = payload["batch_idx"][center].long()
    displacement = (
        payload["positions"][neighbor] - payload["positions"][center]
    )
    displacement = displacement + torch.einsum(
        "ei,eij->ej", images.float(), payload["cells"][edge_batch]
    )

    graphs = []
    for graph_idx in range(len(atoms_list)):
        atom_start = payload["atom_offsets"][graph_idx]
        edge_ids = torch.where(edge_batch == graph_idx)[0]
        graphs.append(
            (
                center[edge_ids] - atom_start,
                neighbor[edge_ids] - atom_start,
                displacement[edge_ids],
                images[edge_ids],
            )
        )
    return graphs if batched else graphs[0]


def _compute_bond_cosines(edges):
    """Compute line-graph angle cosines from bond displacement vectors."""
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    return {"h": torch.clamp(cosine, -1, 1)}


class TensorGraphBuilder:
    """Build ALIGNN DGL graphs directly from GPU-resident positions."""

    def __init__(
        self,
        *,
        cell: Tensor,
        pbc: Tensor,
        node_features: Tensor,
        cutoff: float,
        cutoff_extra: float = 0.5,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = _cuda_device(device)
        self.cell = cell.to(device=self.device, dtype=torch.float32)
        self.inverse_cell = torch.linalg.inv(self.cell)
        self.pbc = pbc.to(device=self.device, dtype=torch.bool)
        self.node_features = node_features.to(
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        self.cutoff = float(cutoff)
        self.cutoff_extra = float(cutoff_extra)
        if self.cutoff_extra <= 0:
            raise ValueError("cutoff_extra must be positive")
        self.n_atoms = int(self.node_features.shape[0])
        self.batch_idx = torch.zeros(
            self.n_atoms,
            dtype=torch.int32,
            device=self.device,
        )
        self.volume = torch.abs(torch.det(self.cell))

    @torch.no_grad()
    def _edge_tensors(
        self,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return exact edge tensors and wrapped fractional coordinates."""
        if positions.shape != (self.n_atoms, 3):
            raise ValueError(
                "positions must have shape "
                f"({self.n_atoms}, 3), got {tuple(positions.shape)}"
            )
        positions = positions.to(device=self.device, dtype=torch.float32)
        frac = (positions @ self.inverse_cell) % 1.0
        wrapped_positions = frac @ self.cell

        graph_cutoff = self.cutoff
        while True:
            center, neighbor, images, _ = neighbor_list_nvidia(
                wrapped_positions,
                self.cell.unsqueeze(0),
                graph_cutoff,
                self.batch_idx,
                self.pbc.unsqueeze(0),
                device=self.device,
            )
            counts = torch.bincount(center, minlength=self.n_atoms)
            if torch.all(counts > 0):
                break
            graph_cutoff += self.cutoff_extra

        displacement = wrapped_positions[neighbor] - wrapped_positions[center]
        displacement = displacement + images.to(torch.float32) @ self.cell
        return center, neighbor, displacement, images, frac

    @torch.no_grad()
    def build_tensors(
        self,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build exact atom and line-graph index tensors directly on CUDA.

        Returns ``(source, destination, displacement, line_source,
        line_destination)``. The line-graph indices have the same directed
        ``a -> b -> c`` semantics as ``DGLGraph.line_graph``.
        """
        center, neighbor, displacement, images, _ = self._edge_tensors(
            positions
        )
        edge_ids = torch.arange(
            center.shape[0], dtype=torch.int64, device=self.device
        )
        max_image = 10
        radix = 2 * max_image + 1
        images_long = images.long()
        edge_keys = (
            (center * self.n_atoms + neighbor) * radix**3
            + (images_long[:, 0] + max_image) * radix * radix
            + (images_long[:, 1] + max_image) * radix
            + images_long[:, 2]
            + max_image
        )
        reverse_keys = (
            (neighbor * self.n_atoms + center) * radix**3
            + (-images_long[:, 0] + max_image) * radix * radix
            + (-images_long[:, 1] + max_image) * radix
            - images_long[:, 2]
            + max_image
        )
        sorted_keys, order = torch.sort(edge_keys)
        reverse_positions = torch.searchsorted(sorted_keys, reverse_keys)
        reverse_positions = reverse_positions.clamp(max=edge_ids.numel() - 1)
        reverse_edges = order[reverse_positions]

        counts = torch.bincount(center, minlength=self.n_atoms)
        offsets = torch.zeros(
            self.n_atoms + 1, dtype=torch.int64, device=self.device
        )
        torch.cumsum(counts, dim=0, out=offsets[1:])
        pair_offsets = torch.zeros_like(offsets)
        torch.cumsum(counts * counts, dim=0, out=pair_offsets[1:])
        triplet_count = int(pair_offsets[-1].item())
        triplet_ids = torch.arange(
            triplet_count, dtype=torch.int64, device=self.device
        )
        atoms = torch.searchsorted(
            pair_offsets, triplet_ids, right=True
        ) - 1
        local_ids = triplet_ids - pair_offsets[atoms]
        degrees = counts[atoms]
        incoming_slots = local_ids // degrees
        outgoing_slots = local_ids % degrees
        incoming_outgoing_edges = offsets[atoms] + incoming_slots
        line_source = reverse_edges[incoming_outgoing_edges]
        line_destination = offsets[atoms] + outgoing_slots
        return (
            center,
            neighbor,
            displacement,
            line_source,
            line_destination,
        )

    @torch.no_grad()
    def build(self, positions: Tensor) -> tuple[dgl.DGLGraph, dgl.DGLGraph]:
        """Construct an atom graph and line graph without a CPU round-trip."""
        center, neighbor, displacement, images, frac = self._edge_tensors(
            positions
        )
        graph = dgl.graph(
            (center, neighbor),
            num_nodes=self.n_atoms,
            device=self.device,
        )
        graph.ndata["atom_features"] = self.node_features
        graph.ndata["V"] = self.volume.expand(self.n_atoms)
        graph.ndata["frac_coords"] = frac
        graph.edata["r"] = displacement
        graph.edata["images"] = images.to(torch.get_default_dtype())

        line_graph = graph.line_graph(shared=True)
        line_graph.apply_edges(_compute_bond_cosines)
        return graph, line_graph
