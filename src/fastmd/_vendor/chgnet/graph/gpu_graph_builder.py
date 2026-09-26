from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from fastmd._vendor.chgnet._nvtx import nvtx_range
from fastmd._vendor.chgnet.graph.crystalgraph import CrystalGraph

try:
    from nvalchemiops.neighborlist.neighbor_utils import estimate_max_neighbors
    from nvalchemiops.neighborlist.neighborlist import neighbor_list as _nvidia_nl

    op_available = True
except ImportError:
    op_available = False

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    triton = None
    tl = None
    _triton_available = False

# Static upper bound on periodic-image shift magnitude, used to size sort keys
# without a .item() D2H sync. |shift| <= MAX_IMAGE holds whenever the smallest
# cell dimension >= cutoff / MAX_IMAGE, i.e. always for physical cells.
MAX_IMAGE = 10
_SPATIAL_DIM = 3
_MATRIX_RANK = 2


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "off", "no"}


_LINE_GRAPH_TRITON = _env_flag("CHGNET_LINE_GRAPH_TRITON", default=True)
_LINE_GRAPH_TRITON_BLOCK = int(os.getenv("CHGNET_LINE_GRAPH_TRITON_BLOCK", "256"))


def _cuda_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(device)
    return device


def _require_nvalchemiops() -> None:
    if not op_available:
        raise RuntimeError(
            "GPU graph construction requires the optional dependencies. "
            "Install them with `pip install 'chgnet[gpu-opt]'`."
        )


def _right_matmul_3x3(points: Tensor, matrix: Tensor) -> Tensor:
    """Multiply [N, 3] by [3, 3] without TF32 coordinate rounding."""
    return torch.stack(
        (
            points[:, 0] * matrix[0, 0]
            + points[:, 1] * matrix[1, 0]
            + points[:, 2] * matrix[2, 0],
            points[:, 0] * matrix[0, 1]
            + points[:, 1] * matrix[1, 1]
            + points[:, 2] * matrix[2, 1],
            points[:, 0] * matrix[0, 2]
            + points[:, 1] * matrix[1, 2]
            + points[:, 2] * matrix[2, 2],
        ),
        dim=1,
    )


def _wrap_fractional_coordinates(frac: Tensor, pbc: Tensor) -> Tensor:
    """Wrap fractional coordinates only along periodic axes."""
    return torch.where(pbc, frac.remainder(1.0), frac)


@torch.no_grad()
def neighbor_list_nvidia(
    positions: Tensor,
    cell: Tensor,
    cutoff: float,
    batch_idx: Tensor,
    pbc: Tensor,
    device: torch.device | str = "cuda",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a periodic neighbor list on a CUDA device."""
    with nvtx_range("neighbor_list_nvidia"):
        _require_nvalchemiops()
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

        _nvidia_nl(
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
        neigh_idx = torch.arange(buffer_max, device=device).unsqueeze(0)
        valid = neigh_idx < num_neighbors.unsqueeze(1)
        center = atom_idx.expand(-1, buffer_max)[valid].long()
        neighbor = neighbor_matrix[valid].long()
        images = neighbor_shifts[valid]

        edge_batch = batch_idx[center].long()
        diff = positions[neighbor] - positions[center]
        diff = diff + torch.einsum("ei,eij->ej", images.float(), cell[edge_batch])
        dist = diff.norm(dim=1)
        mask = dist < cutoff
        center = center[mask]
        neighbor = neighbor[mask]
        images = images[mask]
        dist = dist[mask]

        img = images.long()
        # Static image bound avoids a .item() D2H sync. |shift| <= MAX_IMAGE
        # holds when the smallest cell dimension >= cutoff/MAX_IMAGE.
        max_img = MAX_IMAGE
        radix = 2 * max_img + 1
        order = (
            batch_idx[center].long() * n_atoms * n_atoms * radix**3
            + center * n_atoms * radix**3
            + neighbor * radix**3
            + (img[:, 0] + max_img) * radix * radix
            + (img[:, 1] + max_img) * radix
            + (img[:, 2] + max_img)
        ).argsort()
        return center[order], neighbor[order], images[order], dist[order]


def _build_d2u(
    center: Tensor,
    neighbor: Tensor,
    images: Tensor,
    n_atoms: int,
) -> tuple[Tensor, Tensor]:
    with nvtx_range("_build_d2u"):
        device = center.device
        n_edges = center.shape[0]
        img = images.long()
        max_img = MAX_IMAGE  # static bound -> no .item() D2H sync
        radix = 2 * max_img + 1
        volume = radix * radix * radix
        forward_key = (
            (center * n_atoms + neighbor) * volume
            + (img[:, 0] + max_img) * radix * radix
            + (img[:, 1] + max_img) * radix
            + (img[:, 2] + max_img)
        )
        reverse_key = (
            (neighbor * n_atoms + center) * volume
            + (-img[:, 0] + max_img) * radix * radix
            + (-img[:, 1] + max_img) * radix
            + (-img[:, 2] + max_img)
        )
        sorted_key, sort_idx = forward_key.sort()
        pair_idx = sort_idx[
            torch.searchsorted(sorted_key, reverse_key).clamp(max=n_edges - 1)
        ]
        edge_ids = torch.arange(n_edges, device=device)
        uvals, directed2undirected = torch.unique(
            torch.minimum(edge_ids, pair_idx),
            return_inverse=True,
        )
        undirected2directed = torch.full(
            (
                uvals.shape[0],
            ),  # = num unique undirected edges; shape read, no .item() D2H sync
            n_edges,
            dtype=torch.int64,
            device=device,
        )
        undirected2directed.scatter_reduce_(
            0,
            directed2undirected.long(),
            edge_ids,
            reduce="amin",
            include_self=False,
        )
        return directed2undirected.int(), undirected2directed.int()


if _triton_available:

    @triton.jit
    def _bond_graph_materialize_kernel(  # noqa: ANN202
        counts,
        offsets,
        pair_offsets,
        short_de,
        short_ude,
        out,
        TOTAL_PAIRS,  # noqa: N803
        N_ATOMS: tl.constexpr,  # noqa: N803
        LOG_N: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < TOTAL_PAIRS

        lo = tl.full((BLOCK,), 0, dtype=tl.int64)
        hi = tl.full((BLOCK,), N_ATOMS - 1, dtype=tl.int64)
        for _ in tl.static_range(0, LOG_N):
            mid = (lo + hi + 1) // 2
            mid_offsets = tl.load(pair_offsets + mid)
            take_upper = mid_offsets <= rows
            lo = tl.where(take_upper, mid, lo)
            hi = tl.where(take_upper, hi, mid - 1)

        atom = lo
        count = tl.load(counts + atom)
        atom_pair_offset = tl.load(pair_offsets + atom)
        local_idx = rows - atom_pair_offset
        denom = tl.maximum(count - 1, 1)
        first_in_group = local_idx // denom
        second_raw = local_idx - first_in_group * denom
        second_in_group = second_raw + (second_raw >= first_in_group)
        atom_offset = tl.load(offsets + atom)
        first_pos = atom_offset + first_in_group
        second_pos = atom_offset + second_in_group

        first_ude = tl.load(short_ude + first_pos, mask=mask, other=0)
        first_de = tl.load(short_de + first_pos, mask=mask, other=0)
        second_ude = tl.load(short_ude + second_pos, mask=mask, other=0)
        second_de = tl.load(short_de + second_pos, mask=mask, other=0)
        base = rows * 5
        tl.store(out + base + 0, atom, mask=mask)
        tl.store(out + base + 1, first_ude, mask=mask)
        tl.store(out + base + 2, first_de, mask=mask)
        tl.store(out + base + 3, second_ude, mask=mask)
        tl.store(out + base + 4, second_de, mask=mask)

else:
    _bond_graph_materialize_kernel = None


def _materialize_bond_graph_triton(
    counts: Tensor,
    offsets: Tensor,
    pair_offsets: Tensor,
    short_de: Tensor,
    short_ude: Tensor,
    total_pairs: int,
    n_atoms: int,
) -> Tensor | None:
    if not (_LINE_GRAPH_TRITON and _triton_available and short_de.is_cuda):
        return None
    with nvtx_range("_build_bond_graph_materialize"):
        out = torch.empty((total_pairs, 5), dtype=torch.int32, device=short_de.device)
        block = _LINE_GRAPH_TRITON_BLOCK
        grid = (triton.cdiv(total_pairs, block),)
        _bond_graph_materialize_kernel[grid](
            counts,
            offsets,
            pair_offsets,
            short_de,
            short_ude,
            out,
            TOTAL_PAIRS=total_pairs,
            N_ATOMS=n_atoms,
            LOG_N=max(1, (n_atoms - 1).bit_length()),
            BLOCK=block,
            num_warps=4,
        )
        return out


def _build_bond_graph(
    center: Tensor,
    directed2undirected: Tensor,
    dist: Tensor,
    n_atoms: int,
    cutoff: float,
) -> Tensor:
    with nvtx_range("_build_bond_graph"):
        device = center.device
        short_de = torch.where(dist < cutoff)[0]
        if short_de.numel() == 0:
            return torch.empty((0, 5), dtype=torch.int32, device=device)

        sorted_center, order = center[short_de].sort(stable=True)
        short_de = short_de[order]
        short_ude = directed2undirected[short_de].long()
        counts = torch.bincount(sorted_center.int(), minlength=n_atoms).long()
        pair_counts = counts * (counts - 1)
        total_pairs = int(pair_counts.sum().item())
        if total_pairs == 0:
            return torch.empty((0, 5), dtype=torch.int32, device=device)

        offsets = torch.zeros(n_atoms, dtype=torch.long, device=device)
        offsets[1:] = counts[:-1].cumsum(0)
        pair_offsets = torch.zeros(n_atoms, dtype=torch.long, device=device)
        pair_offsets[1:] = pair_counts[:-1].cumsum(0)
        triton_bond_graph = _materialize_bond_graph_triton(
            counts,
            offsets,
            pair_offsets,
            short_de,
            short_ude,
            total_pairs,
            n_atoms,
        )
        if triton_bond_graph is not None:
            return triton_bond_graph

        pair_atom = torch.repeat_interleave(
            torch.arange(n_atoms, device=device), pair_counts
        )
        local_idx = torch.arange(total_pairs, device=device) - pair_offsets[pair_atom]
        group_size = counts[pair_atom]
        first_in_group = local_idx // (group_size - 1)
        second_raw = local_idx % (group_size - 1)
        second_in_group = second_raw + (second_raw >= first_in_group).long()
        first_pos = offsets[pair_atom] + first_in_group
        second_pos = offsets[pair_atom] + second_in_group

        return torch.stack(
            [
                pair_atom.int(),
                short_ude[first_pos].int(),
                short_de[first_pos].int(),
                short_ude[second_pos].int(),
                short_de[second_pos].int(),
            ],
            dim=1,
        )


def _extract_atoms(
    atoms_list: list,
    device: torch.device,
) -> dict[str, Tensor | list]:
    atomic_numbers = []
    frac_coords = []
    positions = []
    cells = []
    pbc = []
    batch_idx = []
    atom_offsets = [0]
    compositions = []

    # Pack one or more ASE Atoms objects into a single batched CUDA payload.
    for graph_idx, atoms in enumerate(atoms_list):
        cell = torch.from_numpy(atoms.get_cell().array.astype(np.float32)).to(device)
        pos = torch.from_numpy(atoms.get_positions().astype(np.float32)).to(device)
        system_pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
        frac = _wrap_fractional_coordinates(
            _right_matmul_3x3(pos, torch.linalg.inv(cell)),
            system_pbc,
        )
        n_atoms = len(atoms)
        atomic_numbers.append(
            torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=device)
        )
        frac_coords.append(frac)
        positions.append(_right_matmul_3x3(frac, cell))
        cells.append(cell)
        pbc.append(system_pbc)
        batch_idx.append(
            torch.full((n_atoms,), graph_idx, dtype=torch.int32, device=device)
        )
        atom_offsets.append(atom_offsets[-1] + n_atoms)
        compositions.append(atoms.get_chemical_formula())

    return {
        "atomic_numbers": torch.cat(atomic_numbers),
        # torch.cat runs under the public entry point's no-grad context. Mark the
        # concatenated coordinates as a leaf so CHGNet can differentiate forces.
        "frac_coords": torch.cat(frac_coords).requires_grad_(),
        "positions": torch.cat(positions),
        "cells": torch.stack(cells),
        "pbc": torch.stack(pbc),
        "batch_idx": torch.cat(batch_idx),
        "atom_offsets": atom_offsets,
        "compositions": compositions,
    }


def _extract_tensor_system(
    positions: Tensor,
    cell: Tensor,
    atomic_numbers: Tensor,
    pbc: Tensor | None,
    device: torch.device,
    composition: str | None = None,
) -> dict[str, Tensor | list]:
    positions = positions.to(device=device, dtype=torch.float32)
    cell = cell.to(device=device, dtype=torch.float32)
    atomic_numbers = atomic_numbers.to(device=device, dtype=torch.int32)
    if pbc is None:
        pbc = torch.ones(3, dtype=torch.bool, device=device)
    else:
        pbc = pbc.to(device=device, dtype=torch.bool)

    if cell.dim() != _MATRIX_RANK or tuple(cell.shape) != (
        _SPATIAL_DIM,
        _SPATIAL_DIM,
    ):
        raise ValueError(f"cell must have shape (3, 3), got {tuple(cell.shape)}")
    if positions.dim() != _MATRIX_RANK or positions.shape[1] != _SPATIAL_DIM:
        raise ValueError(
            f"positions must have shape (N, 3), got {tuple(positions.shape)}"
        )
    if atomic_numbers.dim() != 1 or atomic_numbers.shape[0] != positions.shape[0]:
        raise ValueError(
            "atomic_numbers must be a 1D tensor with the same length as positions"
        )
    if pbc.dim() != 1 or pbc.shape[0] != _SPATIAL_DIM:
        raise ValueError(f"pbc must have shape (3,), got {tuple(pbc.shape)}")

    frac = _wrap_fractional_coordinates(
        _right_matmul_3x3(positions, torch.linalg.inv(cell)),
        pbc,
    ).requires_grad_()
    n_atoms = int(positions.shape[0])
    return {
        "atomic_numbers": atomic_numbers,
        "frac_coords": frac,
        "positions": _right_matmul_3x3(frac, cell),
        "cells": cell.unsqueeze(0),
        "pbc": pbc.unsqueeze(0),
        "batch_idx": torch.zeros(n_atoms, dtype=torch.int32, device=device),
        "atom_offsets": [0, n_atoms],
        "compositions": [composition or ""],
    }


def _split_graphs(
    payload: dict[str, Tensor | list],
    center: Tensor,
    neighbor: Tensor,
    images: Tensor,
    directed2undirected: Tensor,
    undirected2directed: Tensor,
    bond_graph: Tensor,
    atom_graph_cutoff: float,
    bond_graph_cutoff: float,
) -> list[CrystalGraph]:
    # Single-system fast path (the MD case): atom_start==0 and all edges/triplets
    # belong to graph 0, so global indices ARE local — skip the per-graph
    # where/unique/searchsorted/boolean-mask remapping (~7 D2H syncs/step).
    if len(payload["atom_offsets"]) - 1 == 1:
        return [
            CrystalGraph(
                atomic_number=payload["atomic_numbers"],
                atom_frac_coord=payload["frac_coords"],
                atom_graph=torch.stack([center, neighbor], dim=1).int(),
                neighbor_image=images.float(),
                directed2undirected=directed2undirected.int(),
                undirected2directed=undirected2directed.int(),
                bond_graph=bond_graph.int(),
                lattice=payload["cells"][0],
                graph_id=None,
                mp_id=None,
                composition=payload["compositions"][0],
                atom_graph_cutoff=atom_graph_cutoff,
                bond_graph_cutoff=bond_graph_cutoff,
            )
        ]

    graphs = []
    for graph_idx in range(len(payload["atom_offsets"]) - 1):
        atom_start = payload["atom_offsets"][graph_idx]
        atom_end = payload["atom_offsets"][graph_idx + 1]
        edge_ids = torch.where(payload["batch_idx"][center] == graph_idx)[0].long()
        local_d2u_global = directed2undirected[edge_ids].long()
        unique_ude_global, local_d2u = torch.unique(
            local_d2u_global,
            sorted=True,
            return_inverse=True,
        )
        local_bond_graph = (
            bond_graph[(bond_graph[:, 0] >= atom_start) & (bond_graph[:, 0] < atom_end)]
            .clone()
            .long()
        )
        if local_bond_graph.numel() != 0:
            local_bond_graph[:, 0] -= atom_start
            local_bond_graph[:, 1] = torch.searchsorted(
                unique_ude_global,
                local_bond_graph[:, 1].contiguous(),
            )
            local_bond_graph[:, 2] = torch.searchsorted(
                edge_ids,
                local_bond_graph[:, 2].contiguous(),
            )
            local_bond_graph[:, 3] = torch.searchsorted(
                unique_ude_global,
                local_bond_graph[:, 3].contiguous(),
            )
            local_bond_graph[:, 4] = torch.searchsorted(
                edge_ids,
                local_bond_graph[:, 4].contiguous(),
            )

        graphs.append(
            CrystalGraph(
                atomic_number=payload["atomic_numbers"][atom_start:atom_end],
                atom_frac_coord=payload["frac_coords"][atom_start:atom_end],
                atom_graph=torch.stack(
                    [center[edge_ids] - atom_start, neighbor[edge_ids] - atom_start],
                    dim=1,
                ).int(),
                neighbor_image=images[edge_ids].float(),
                directed2undirected=local_d2u.int(),
                undirected2directed=torch.searchsorted(
                    edge_ids,
                    undirected2directed[unique_ude_global].long(),
                ).int(),
                bond_graph=local_bond_graph.int(),
                lattice=payload["cells"][graph_idx],
                graph_id=None,
                mp_id=None,
                composition=payload["compositions"][graph_idx],
                atom_graph_cutoff=atom_graph_cutoff,
                bond_graph_cutoff=bond_graph_cutoff,
            )
        )
    return graphs


def _build_graphs_from_payload(
    payload: dict[str, Tensor | list],
    atom_graph_cutoff: float,
    bond_graph_cutoff: float,
    device: torch.device,
) -> list[CrystalGraph]:
    center, neighbor, images, dist = neighbor_list_nvidia(
        payload["positions"],
        payload["cells"],
        atom_graph_cutoff,
        payload["batch_idx"],
        payload["pbc"],
        device=device,
    )
    directed2undirected, undirected2directed = _build_d2u(
        center,
        neighbor,
        images,
        int(payload["atomic_numbers"].shape[0]),
    )
    bond_graph = _build_bond_graph(
        center,
        directed2undirected,
        dist,
        int(payload["atomic_numbers"].shape[0]),
        bond_graph_cutoff,
    )
    return _split_graphs(
        payload,
        center,
        neighbor,
        images,
        directed2undirected,
        undirected2directed,
        bond_graph,
        atom_graph_cutoff,
        bond_graph_cutoff,
    )


class TensorGraphBuilder:
    """Cached single-system tensor graph builder for fixed-cell GPU MD."""

    def __init__(
        self,
        *,
        cell: Tensor,
        atomic_numbers: Tensor,
        pbc: Tensor | None = None,
        atom_graph_cutoff: float = 6.0,
        bond_graph_cutoff: float = 3.0,
        device: torch.device | str = "cuda",
        composition: str | None = None,
    ) -> None:
        """Initialize a builder for one fixed-cell atomic system."""
        self.device = _cuda_device(device)
        self.cell = cell.to(device=self.device, dtype=torch.float32)
        self.inv_cell = torch.linalg.inv(self.cell)
        self.cells = self.cell.unsqueeze(0)
        self.atomic_numbers = atomic_numbers.to(device=self.device, dtype=torch.int32)
        if pbc is None:
            pbc = torch.ones(3, dtype=torch.bool, device=self.device)
        else:
            pbc = pbc.to(device=self.device, dtype=torch.bool)
        self.pbc = pbc.unsqueeze(0)
        self.n_atoms = int(self.atomic_numbers.shape[0])
        self.batch_idx = torch.zeros(
            self.n_atoms, dtype=torch.int32, device=self.device
        )
        self.atom_offsets = [0, self.n_atoms]
        self.compositions = [composition or ""]
        self.atom_graph_cutoff = float(atom_graph_cutoff)
        self.bond_graph_cutoff = float(bond_graph_cutoff)

        if self.cell.dim() != _MATRIX_RANK or tuple(self.cell.shape) != (
            _SPATIAL_DIM,
            _SPATIAL_DIM,
        ):
            raise ValueError(
                f"cell must have shape (3, 3), got {tuple(self.cell.shape)}"
            )
        if self.pbc.dim() != _MATRIX_RANK or tuple(self.pbc.shape) != (
            1,
            _SPATIAL_DIM,
        ):
            raise ValueError(
                f"pbc must have shape (3,), got {tuple(self.pbc.squeeze(0).shape)}"
            )
        if self.atomic_numbers.dim() != 1:
            raise ValueError("atomic_numbers must be a 1D tensor")

    @torch.no_grad()
    def build(self, positions: Tensor) -> CrystalGraph:
        """Build a crystal graph from Cartesian positions."""
        with nvtx_range("tensors_to_graph_gpu"):
            positions = positions.to(device=self.device, dtype=torch.float32)
            if positions.dim() != _MATRIX_RANK or positions.shape[1] != _SPATIAL_DIM:
                raise ValueError(
                    f"positions must have shape (N, 3), got {tuple(positions.shape)}"
                )
            if positions.shape[0] != self.n_atoms:
                raise ValueError(
                    "positions must have the same length as atomic_numbers "
                    f"({positions.shape[0]} != {self.n_atoms})"
                )
            frac = _wrap_fractional_coordinates(
                _right_matmul_3x3(positions, self.inv_cell),
                self.pbc[0],
            ).requires_grad_()
            payload = {
                "atomic_numbers": self.atomic_numbers,
                "frac_coords": frac,
                "positions": _right_matmul_3x3(frac, self.cell),
                "cells": self.cells,
                "pbc": self.pbc,
                "batch_idx": self.batch_idx,
                "atom_offsets": self.atom_offsets,
                "compositions": self.compositions,
            }
            return _build_graphs_from_payload(
                payload,
                self.atom_graph_cutoff,
                self.bond_graph_cutoff,
                self.device,
            )[0]

    __call__ = build


@torch.no_grad()
def tensors_to_graph_gpu(
    positions: Tensor,
    cell: Tensor,
    atomic_numbers: Tensor,
    pbc: Tensor | None = None,
    atom_graph_cutoff: float = 6.0,
    bond_graph_cutoff: float = 3.0,
    device: torch.device | str = "cuda",
    composition: str | None = None,
) -> CrystalGraph:
    """Build a single-system CrystalGraph directly from CUDA/CPU tensors.

    This is the GPU-resident MD entry point: callers can keep positions, cell,
    and atomic numbers as tensors and avoid the per-step ASE -> NumPy -> CUDA
    packing performed by :func:`atoms_to_graph_gpu`.
    """
    with nvtx_range("tensors_to_graph_gpu"):
        device = _cuda_device(device)
        payload = _extract_tensor_system(
            positions=positions,
            cell=cell,
            atomic_numbers=atomic_numbers,
            pbc=pbc,
            device=device,
            composition=composition,
        )
        return _build_graphs_from_payload(
            payload,
            atom_graph_cutoff,
            bond_graph_cutoff,
            device,
        )[0]


@torch.no_grad()
def atoms_to_graph_gpu(
    atoms,
    atom_graph_cutoff: float = 6.0,
    bond_graph_cutoff: float = 3.0,
    device: torch.device | str = "cuda",
) -> CrystalGraph | list[CrystalGraph]:
    """Build one or more crystal graphs from ASE Atoms on a CUDA device."""
    device = _cuda_device(device)
    batched = not hasattr(atoms, "get_positions")
    atoms_list = list(atoms) if batched else [atoms]
    payload = _extract_atoms(atoms_list, device)

    # NOTE: the per-step isolated-atom check (bincount + .item()) is intentionally
    # omitted here — it forced two D2H syncs every step. Isolated atoms do not
    # occur in condensed-phase MD; validate once at setup if needed.
    graphs = _build_graphs_from_payload(
        payload,
        atom_graph_cutoff,
        bond_graph_cutoff,
        device,
    )
    return graphs if batched else graphs[0]
