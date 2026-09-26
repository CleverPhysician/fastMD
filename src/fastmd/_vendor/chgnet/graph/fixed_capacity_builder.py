"""Capture-safe fixed-capacity graph construction for GPU-resident MD.

The regular GPU graph builder returns ragged edge and triplet tensors.  This
module keeps all allocations static and routes unused rows to dummy sink atoms,
so graph construction can eventually live in the same CUDA Graph as the model
and integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .._nvtx import nvtx_range
from .crystalgraph import CrystalGraph
from .gpu_graph_builder import MAX_IMAGE, _cuda_device, _right_matmul_3x3

try:
    import triton
    import triton.language as tl
    from nvalchemiops.neighborlist.batch_cell_list import (
        allocate_cell_list,
        batch_build_cell_list,
        batch_query_cell_list,
        estimate_batch_cell_list_sizes,
    )
    from nvalchemiops.neighborlist.neighbor_utils import (
        NeighborOverflowError,
        estimate_max_neighbors,
    )

    _fixed_builder_available = True
except ImportError:
    triton = None
    tl = None
    _fixed_builder_available = False

fixed_capacity_available = _fixed_builder_available


if _fixed_builder_available:

    @triton.jit
    def _count_neighbors_kernel(
        positions,
        cell,
        neighbor_matrix,
        neighbor_shifts,
        num_neighbors,
        wrap_delta,
        edge_counts,
        short_counts,
        isolated_atom_count,
        N: tl.constexpr,
        BMAX: tl.constexpr,
        BLOCK: tl.constexpr,
        EDGE_CUTOFF2: tl.constexpr,
        LINE_CUTOFF2: tl.constexpr,
    ):
        atom = tl.program_id(0)
        slots = tl.arange(0, BLOCK)
        slot_mask = slots < BMAX
        count = tl.load(num_neighbors + atom)
        valid_slot = slot_mask & (slots < count)
        neighbor = tl.load(
            neighbor_matrix + atom * BMAX + slots,
            mask=valid_slot,
            other=0,
        )
        neighbor = tl.minimum(neighbor, N - 1)

        px = tl.load(positions + atom * 3)
        py = tl.load(positions + atom * 3 + 1)
        pz = tl.load(positions + atom * 3 + 2)
        nx = tl.load(positions + neighbor * 3, mask=valid_slot, other=0.0)
        ny = tl.load(positions + neighbor * 3 + 1, mask=valid_slot, other=0.0)
        nz = tl.load(positions + neighbor * 3 + 2, mask=valid_slot, other=0.0)

        shift_base = (atom * BMAX + slots) * 3
        sx = tl.load(neighbor_shifts + shift_base, mask=valid_slot, other=0).to(
            tl.float32
        )
        sy = tl.load(neighbor_shifts + shift_base + 1, mask=valid_slot, other=0).to(
            tl.float32
        )
        sz = tl.load(neighbor_shifts + shift_base + 2, mask=valid_slot, other=0).to(
            tl.float32
        )
        sx += tl.load(wrap_delta + neighbor * 3, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3
        )
        sy += tl.load(
            wrap_delta + neighbor * 3 + 1, mask=valid_slot, other=0
        ) - tl.load(wrap_delta + atom * 3 + 1)
        sz += tl.load(
            wrap_delta + neighbor * 3 + 2, mask=valid_slot, other=0
        ) - tl.load(wrap_delta + atom * 3 + 2)
        dx = (
            nx
            - px
            + sx * tl.load(cell)
            + sy * tl.load(cell + 3)
            + sz * tl.load(cell + 6)
        )
        dy = (
            ny
            - py
            + sx * tl.load(cell + 1)
            + sy * tl.load(cell + 4)
            + sz * tl.load(cell + 7)
        )
        dz = (
            nz
            - pz
            + sx * tl.load(cell + 2)
            + sy * tl.load(cell + 5)
            + sz * tl.load(cell + 8)
        )
        dist2 = dx * dx + dy * dy + dz * dz
        edge_valid = valid_slot & (dist2 < EDGE_CUTOFF2)
        short_valid = valid_slot & (dist2 < LINE_CUTOFF2)
        edge_count = tl.sum(edge_valid.to(tl.int32), axis=0)
        tl.store(edge_counts + atom, edge_count)
        tl.store(short_counts + atom, tl.sum(short_valid.to(tl.int32), axis=0))
        tl.atomic_add(isolated_atom_count, 1, mask=edge_count == 0)

    @triton.jit
    def _materialize_neighbors_kernel(
        positions,
        cell,
        neighbor_matrix,
        neighbor_shifts,
        num_neighbors,
        wrap_delta,
        edge_offsets,
        center_out,
        neighbor_out,
        image_out,
        distance_out,
        short_flag_out,
        short_edge_matrix,
        N: tl.constexpr,
        BMAX: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
        EDGE_CUTOFF2: tl.constexpr,
        LINE_CUTOFF2: tl.constexpr,
    ):
        atom = tl.program_id(0)
        slots = tl.arange(0, BLOCK)
        slot_mask = slots < BMAX
        count = tl.load(num_neighbors + atom)
        valid_slot = slot_mask & (slots < count)
        neighbor = tl.load(
            neighbor_matrix + atom * BMAX + slots,
            mask=valid_slot,
            other=0,
        )
        neighbor = tl.minimum(neighbor, N - 1)

        px = tl.load(positions + atom * 3)
        py = tl.load(positions + atom * 3 + 1)
        pz = tl.load(positions + atom * 3 + 2)
        nx = tl.load(positions + neighbor * 3, mask=valid_slot, other=0.0)
        ny = tl.load(positions + neighbor * 3 + 1, mask=valid_slot, other=0.0)
        nz = tl.load(positions + neighbor * 3 + 2, mask=valid_slot, other=0.0)

        shift_base = (atom * BMAX + slots) * 3
        sx_i = tl.load(neighbor_shifts + shift_base, mask=valid_slot, other=0)
        sy_i = tl.load(neighbor_shifts + shift_base + 1, mask=valid_slot, other=0)
        sz_i = tl.load(neighbor_shifts + shift_base + 2, mask=valid_slot, other=0)
        sx_i += tl.load(wrap_delta + neighbor * 3, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3
        )
        sy_i += tl.load(
            wrap_delta + neighbor * 3 + 1, mask=valid_slot, other=0
        ) - tl.load(wrap_delta + atom * 3 + 1)
        sz_i += tl.load(
            wrap_delta + neighbor * 3 + 2, mask=valid_slot, other=0
        ) - tl.load(wrap_delta + atom * 3 + 2)
        sx = sx_i.to(tl.float32)
        sy = sy_i.to(tl.float32)
        sz = sz_i.to(tl.float32)
        dx = (
            nx
            - px
            + sx * tl.load(cell)
            + sy * tl.load(cell + 3)
            + sz * tl.load(cell + 6)
        )
        dy = (
            ny
            - py
            + sx * tl.load(cell + 1)
            + sy * tl.load(cell + 4)
            + sz * tl.load(cell + 7)
        )
        dz = (
            nz
            - pz
            + sx * tl.load(cell + 2)
            + sy * tl.load(cell + 5)
            + sz * tl.load(cell + 8)
        )
        dist2 = dx * dx + dy * dy + dz * dz
        edge_valid = valid_slot & (dist2 < EDGE_CUTOFF2)
        short_valid = valid_slot & (dist2 < LINE_CUTOFF2)

        edge_rank = tl.cumsum(edge_valid.to(tl.int32), axis=0) - 1
        edge_index = tl.load(edge_offsets + atom) + edge_rank
        edge_store = edge_valid & (edge_index < E_CAP)
        tl.store(center_out + edge_index, atom, mask=edge_store)
        tl.store(neighbor_out + edge_index, neighbor, mask=edge_store)
        tl.store(image_out + edge_index * 3, sx_i, mask=edge_store)
        tl.store(image_out + edge_index * 3 + 1, sy_i, mask=edge_store)
        tl.store(image_out + edge_index * 3 + 2, sz_i, mask=edge_store)
        tl.store(distance_out + edge_index, tl.sqrt(dist2), mask=edge_store)
        tl.store(short_flag_out + edge_index, short_valid, mask=edge_store)

        short_rank = tl.cumsum(short_valid.to(tl.int32), axis=0) - 1
        short_store = short_valid & (short_rank < BMAX) & (edge_index < E_CAP)
        tl.store(
            short_edge_matrix + atom * BMAX + short_rank,
            edge_index,
            mask=short_store,
        )

    @triton.jit
    def _pad_edges_kernel(
        total_edges,
        center,
        neighbor,
        images,
        distances,
        short_flags,
        N: tl.constexpr,
        N_DUMMY: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total = tl.load(total_edges)
        total_clamped = tl.minimum(total, E_CAP)
        mask = (rows < E_CAP) & (rows >= total_clamped)
        real_u = total_clamped // 2
        n_pad_u = E_CAP // 2 - real_u
        pad_pair = (rows - total_clamped) // 2
        sink = N + (pad_pair * N_DUMMY) // tl.maximum(n_pad_u, 1)
        tl.store(center + rows, sink, mask=mask)
        tl.store(neighbor + rows, sink, mask=mask)
        tl.store(images + rows * 3, 3, mask=mask)
        tl.store(images + rows * 3 + 1, 3, mask=mask)
        tl.store(images + rows * 3 + 2, 3, mask=mask)
        tl.store(distances + rows, 1.0e6, mask=mask)
        tl.store(short_flags + rows, 0, mask=mask)


@dataclass
class FixedCapacityGraphStatus:
    edge_count: Tensor
    undirected_count: Tensor
    triplet_count: Tensor
    overflow: Tensor
    invalid_pairs: Tensor
    image_bounds_valid: Tensor
    candidate_valid: Tensor
    max_displacement: Tensor
    isolated_atom_count: Tensor


@dataclass(frozen=True)
class StaticizationInvariantReport:
    edge_count: int
    undirected_count: int
    triplet_count: int
    real_edges_isolated: bool
    padding_edges_isolated: bool
    real_triplets_isolated: bool
    padding_triplets_isolated: bool
    reverse_pairs_valid: bool
    sorted_segments_valid: bool
    reference_exact: bool | None


@torch.no_grad()
def validate_staticization_invariants(
    graph: CrystalGraph,
    status: FixedCapacityGraphStatus,
    *,
    n_real: int,
    reference: CrystalGraph | None = None,
) -> StaticizationInvariantReport:
    """Synchronously validate the sink-padding contract for tests/artifacts."""
    edge_count = int(status.edge_count.item())
    undirected_count = int(status.undirected_count.item())
    triplet_count = int(status.triplet_count.item())
    if bool(status.overflow.item()):
        raise AssertionError("cannot validate an overflowed fixed-capacity graph")

    atom_graph = graph.atom_graph.long()
    bond_graph = graph.bond_graph.long()
    d2u = graph.directed2undirected.long()
    u2d = graph.undirected2directed.long()
    real_edges_isolated = bool((atom_graph[:edge_count] < n_real).all().item())
    padding_edges_isolated = bool((atom_graph[edge_count:] >= n_real).all().item())
    real_triplets_isolated = bool(
        (
            (bond_graph[:triplet_count, 0] < n_real)
            & (bond_graph[:triplet_count, 1] < undirected_count)
            & (bond_graph[:triplet_count, 2] < edge_count)
            & (bond_graph[:triplet_count, 3] < undirected_count)
            & (bond_graph[:triplet_count, 4] < edge_count)
        )
        .all()
        .item()
    )
    padding_triplets_isolated = bool(
        (
            (bond_graph[triplet_count:, 0] >= n_real)
            & (bond_graph[triplet_count:, 1] >= undirected_count)
            & (bond_graph[triplet_count:, 2] >= edge_count)
            & (bond_graph[triplet_count:, 3] >= undirected_count)
            & (bond_graph[triplet_count:, 4] >= edge_count)
        )
        .all()
        .item()
    )

    real_pair_rows = torch.argsort(d2u[:edge_count]).reshape(undirected_count, 2)
    first = atom_graph[real_pair_rows[:, 0]]
    second = atom_graph[real_pair_rows[:, 1]]
    first_image = graph.neighbor_image[real_pair_rows[:, 0]].long()
    second_image = graph.neighbor_image[real_pair_rows[:, 1]].long()
    reverse_pairs_valid = bool(
        (
            (first[:, 0] == second[:, 1])
            & (first[:, 1] == second[:, 0])
            & (first_image == -second_image).all(dim=1)
            & (
                d2u[u2d[:undirected_count]]
                == torch.arange(undirected_count, device=d2u.device)
            ).all()
        )
        .all()
        .item()
    )
    edges_sorted = edge_count < 2 or bool(
        (atom_graph[1:edge_count, 0] >= atom_graph[: edge_count - 1, 0]).all().item()
    )
    triplets_sorted = triplet_count < 2 or bool(
        (bond_graph[1:triplet_count, 0] >= bond_graph[: triplet_count - 1, 0])
        .all()
        .item()
    )
    sorted_segments_valid = edges_sorted and triplets_sorted

    reference_exact = None
    if reference is not None:
        reference_exact = all(
            (
                torch.equal(
                    graph.atomic_number[:n_real],
                    reference.atomic_number,
                ),
                torch.equal(
                    graph.atom_frac_coord[:n_real],
                    reference.atom_frac_coord,
                ),
                torch.equal(graph.atom_graph[:edge_count], reference.atom_graph),
                torch.equal(
                    graph.neighbor_image[:edge_count],
                    reference.neighbor_image,
                ),
                torch.equal(
                    graph.directed2undirected[:edge_count],
                    reference.directed2undirected,
                ),
                torch.equal(
                    graph.undirected2directed[:undirected_count],
                    reference.undirected2directed,
                ),
                torch.equal(
                    graph.bond_graph[:triplet_count],
                    reference.bond_graph,
                ),
                torch.equal(graph.lattice, reference.lattice),
            )
        )

    report = StaticizationInvariantReport(
        edge_count=edge_count,
        undirected_count=undirected_count,
        triplet_count=triplet_count,
        real_edges_isolated=real_edges_isolated,
        padding_edges_isolated=padding_edges_isolated,
        real_triplets_isolated=real_triplets_isolated,
        padding_triplets_isolated=padding_triplets_isolated,
        reverse_pairs_valid=reverse_pairs_valid,
        sorted_segments_valid=sorted_segments_valid,
        reference_exact=reference_exact,
    )
    checks = (
        real_edges_isolated,
        padding_edges_isolated,
        real_triplets_isolated,
        padding_triplets_isolated,
        reverse_pairs_valid,
        sorted_segments_valid,
        reference_exact is not False,
    )
    if not all(checks):
        raise AssertionError(f"staticization invariant failure: {report}")
    return report


class FixedCapacityGraphBuilder:
    """Build a single-system sink-padded ``CrystalGraph`` with static shapes."""

    def __init__(
        self,
        *,
        cell: Tensor,
        atomic_numbers: Tensor,
        u_capacity: int,
        t_capacity: int,
        pbc: Tensor | None = None,
        atom_graph_cutoff: float = 6.0,
        bond_graph_cutoff: float = 3.0,
        n_dummy: int = 64,
        device: torch.device | str = "cuda",
        composition: str | None = None,
        candidate_cutoff: float | None = None,
    ) -> None:
        if not _fixed_builder_available:
            raise RuntimeError(
                "fixed-capacity graph builder requires Triton and nvalchemiops"
            )
        self.device = _cuda_device(device)
        self.cell = cell.to(device=self.device, dtype=torch.float32)
        if tuple(self.cell.shape) != (3, 3):
            raise ValueError(
                f"cell must have shape (3, 3), got {tuple(self.cell.shape)}"
            )
        self.cells = self.cell.unsqueeze(0)
        self.inv_cell = torch.linalg.inv(self.cell)
        self.atomic_numbers_real = atomic_numbers.to(
            device=self.device, dtype=torch.int32
        )
        if self.atomic_numbers_real.ndim != 1 or self.atomic_numbers_real.numel() == 0:
            raise ValueError("atomic_numbers must be a non-empty 1D tensor")
        self.n_real = int(self.atomic_numbers_real.shape[0])
        self.n_dummy = int(n_dummy)
        self.u_capacity = int(u_capacity)
        self.e_capacity = 2 * self.u_capacity
        self.t_capacity = int(t_capacity)
        self.atom_graph_cutoff = float(atom_graph_cutoff)
        self.bond_graph_cutoff = float(bond_graph_cutoff)
        self.candidate_cutoff = float(
            self.atom_graph_cutoff + 1.0e-6
            if candidate_cutoff is None
            else candidate_cutoff
        )
        self.skin = self.candidate_cutoff - self.atom_graph_cutoff
        self.composition = composition or ""
        if self.u_capacity <= 0 or self.t_capacity <= 0 or self.n_dummy <= 0:
            raise ValueError("capacities and n_dummy must be positive")
        if self.bond_graph_cutoff > self.atom_graph_cutoff:
            raise ValueError("line graph cutoff must not exceed atom graph cutoff")
        if self.skin <= 0:
            raise ValueError("candidate_cutoff must exceed atom_graph_cutoff")

        if pbc is None:
            pbc = torch.ones(3, dtype=torch.bool, device=self.device)
        if pbc.numel() != 3:
            raise ValueError(
                f"pbc must contain three values, got shape {tuple(pbc.shape)}"
            )
        self.pbc = pbc.to(device=self.device, dtype=torch.bool).reshape(1, 3)
        self.batch_idx = torch.zeros(self.n_real, dtype=torch.int32, device=self.device)
        self.buffer_max = int(
            estimate_max_neighbors(cutoff=self.candidate_cutoff, safety_factor=2.0)
        )
        self._slot_block = triton.next_power_of_2(self.buffer_max)

        max_total_cells, neighbor_radius = estimate_batch_cell_list_sizes(
            self.cells,
            self.pbc,
            self.candidate_cutoff,
        )
        self.cell_list_cache = allocate_cell_list(
            self.n_real,
            max_total_cells,
            neighbor_radius,
            self.device,
        )
        self.neighbor_matrix = torch.full(
            (self.n_real, self.buffer_max),
            self.n_real,
            dtype=torch.int32,
            device=self.device,
        )
        self.neighbor_shifts = torch.zeros(
            (self.n_real, self.buffer_max, 3),
            dtype=torch.int32,
            device=self.device,
        )
        self.num_neighbors = torch.zeros(
            self.n_real, dtype=torch.int32, device=self.device
        )
        self.reference_positions = torch.zeros(
            (self.n_real, 3), dtype=torch.float32, device=self.device
        )
        self.reference_wraps = torch.zeros(
            (self.n_real, 3), dtype=torch.int32, device=self.device
        )
        self.wrap_delta = torch.zeros_like(self.reference_wraps)
        self._candidate_initialized = False

        self.edge_counts = torch.zeros(
            self.n_real, dtype=torch.int64, device=self.device
        )
        self.short_counts = torch.zeros_like(self.edge_counts)
        self.edge_offsets = torch.zeros(
            self.n_real + 1, dtype=torch.int64, device=self.device
        )
        self.short_offsets = torch.zeros_like(self.edge_offsets)
        self.pair_offsets = torch.zeros_like(self.edge_offsets)
        self.short_edge_matrix = torch.zeros(
            (self.n_real, self.buffer_max), dtype=torch.int32, device=self.device
        )

        self.center = torch.empty(
            self.e_capacity, dtype=torch.int32, device=self.device
        )
        self.neighbor = torch.empty_like(self.center)
        self.images = torch.empty(
            (self.e_capacity, 3), dtype=torch.int32, device=self.device
        )
        self.distances = torch.empty(
            self.e_capacity, dtype=torch.float32, device=self.device
        )
        self.short_flags = torch.empty(
            self.e_capacity, dtype=torch.bool, device=self.device
        )
        self.directed2undirected = torch.empty_like(self.center)
        self.undirected2directed = torch.empty(
            self.u_capacity, dtype=torch.int32, device=self.device
        )
        self.bond_graph = torch.empty(
            (self.t_capacity, 5), dtype=torch.int32, device=self.device
        )

        atomic_numbers_all = torch.cat(
            [
                self.atomic_numbers_real,
                torch.ones(self.n_dummy, dtype=torch.int32, device=self.device),
            ]
        )
        self.frac_coords = torch.zeros(
            (self.n_real + self.n_dummy, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.graph = CrystalGraph(
            graph_id=None,
            mp_id=None,
            composition=self.composition,
            atomic_number=atomic_numbers_all,
            atom_frac_coord=self.frac_coords,
            lattice=self.cell,
            neighbor_image=self.images.float(),
            atom_graph=torch.stack([self.center, self.neighbor], dim=1),
            atom_graph_cutoff=self.atom_graph_cutoff,
            bond_graph=self.bond_graph,
            bond_graph_cutoff=self.bond_graph_cutoff,
            directed2undirected=self.directed2undirected,
            undirected2directed=self.undirected2directed,
        )

        self.edge_ids = torch.arange(
            self.e_capacity, device=self.device, dtype=torch.int64
        )
        self.u_ids = torch.arange(
            self.u_capacity, device=self.device, dtype=torch.int64
        )
        self.triplet_ids = torch.arange(
            self.t_capacity, device=self.device, dtype=torch.int64
        )
        self._u2d_scatter = torch.empty(
            self.u_capacity + 1, dtype=torch.int64, device=self.device
        )
        self._short_scatter = torch.empty(
            self.n_real * self.buffer_max + 1,
            dtype=torch.int64,
            device=self.device,
        )
        self._overflow = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._invalid_pairs = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._image_bounds_valid = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._candidate_valid = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._candidate_capacity_valid = torch.zeros(
            1, dtype=torch.bool, device=self.device
        )
        self._max_displacement = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._isolated_atom_count = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )

        # An overflow is observed only after the captured model evaluation. Keep
        # that evaluation memory-safe by substituting a valid all-sink topology
        # on device; the transaction runner then rolls back and grows capacity.
        safe_u = self.edge_ids // 2
        safe_dummy = self.n_real + safe_u * self.n_dummy // self.u_capacity
        self._overflow_center = safe_dummy.int()
        self._overflow_neighbor = self._overflow_center.clone()
        self._overflow_images = torch.zeros(
            (self.e_capacity, 3), dtype=torch.int32, device=self.device
        )
        image_sign = torch.where(
            self.edge_ids.remainder(2) == 0,
            torch.ones_like(self.edge_ids),
            -torch.ones_like(self.edge_ids),
        ).int()
        self._overflow_images.copy_(image_sign[:, None].expand(-1, 3) * 3)
        self._overflow_d2u = safe_u.int()
        self._overflow_u2d = (2 * self.u_ids).int()
        line_dummy = self.triplet_ids * self.n_dummy // self.t_capacity
        line_u = (line_dummy * self.u_capacity + self.n_dummy - 1) // self.n_dummy
        line_u = line_u.clamp(max=self.u_capacity - 1)
        line_de = 2 * line_u
        line_atom = self._overflow_center[line_de]
        self._overflow_bond_graph = torch.stack(
            [line_atom, line_u, line_de, line_u, line_de], dim=1
        ).int()

    def _build_candidates(self, wrapped_positions: Tensor) -> None:
        self.neighbor_matrix.fill_(self.n_real)
        self.neighbor_shifts.zero_()
        self.num_neighbors.zero_()
        batch_build_cell_list(
            wrapped_positions,
            self.candidate_cutoff,
            self.cells,
            self.pbc,
            self.batch_idx,
            *self.cell_list_cache,
        )
        batch_query_cell_list(
            wrapped_positions,
            self.cells,
            self.pbc,
            self.candidate_cutoff,
            self.batch_idx,
            *self.cell_list_cache,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            False,
        )
        self._candidate_capacity_valid.copy_(
            (self.num_neighbors <= self.buffer_max).all().reshape(1)
        )

    def assert_candidate_capacity(self) -> None:
        """Raise on the slow path if nvalchemi truncated a neighbor row."""
        if bool(self._candidate_capacity_valid.item()):
            return
        raise NeighborOverflowError(
            self.buffer_max, int(self.num_neighbors.max().item())
        )

    @torch.no_grad()
    def refresh_candidates(self, positions: Tensor) -> None:
        """Refresh the slow-path cell list used by subsequent exact re-filter steps."""
        positions = positions.to(device=self.device, dtype=torch.float32)
        frac_unwrapped = _right_matmul_3x3(positions, self.inv_cell)
        wraps = torch.where(
            self.pbc[0],
            torch.floor(frac_unwrapped),
            torch.zeros_like(frac_unwrapped),
        ).to(torch.int32)
        wrapped = _right_matmul_3x3(frac_unwrapped - wraps.float(), self.cell)
        self.reference_positions.copy_(positions)
        self.reference_wraps.copy_(wraps)
        self.wrap_delta.zero_()
        self._build_candidates(wrapped)
        self._candidate_initialized = True

    def _materialize_edges(self, wrapped_positions: Tensor) -> None:
        self._isolated_atom_count.zero_()
        _count_neighbors_kernel[(self.n_real,)](
            wrapped_positions,
            self.cell,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            self.wrap_delta,
            self.edge_counts,
            self.short_counts,
            self._isolated_atom_count,
            N=self.n_real,
            BMAX=self.buffer_max,
            BLOCK=self._slot_block,
            EDGE_CUTOFF2=self.atom_graph_cutoff * self.atom_graph_cutoff,
            LINE_CUTOFF2=self.bond_graph_cutoff * self.bond_graph_cutoff,
            num_warps=8,
        )
        self.edge_offsets[0].zero_()
        torch.cumsum(self.edge_counts, dim=0, out=self.edge_offsets[1:])
        self.short_edge_matrix.zero_()
        _materialize_neighbors_kernel[(self.n_real,)](
            wrapped_positions,
            self.cell,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            self.wrap_delta,
            self.edge_offsets,
            self.center,
            self.neighbor,
            self.images,
            self.distances,
            self.short_flags,
            self.short_edge_matrix,
            N=self.n_real,
            BMAX=self.buffer_max,
            E_CAP=self.e_capacity,
            BLOCK=self._slot_block,
            EDGE_CUTOFF2=self.atom_graph_cutoff * self.atom_graph_cutoff,
            LINE_CUTOFF2=self.bond_graph_cutoff * self.bond_graph_cutoff,
            num_warps=8,
        )
        _pad_edges_kernel[(triton.cdiv(self.e_capacity, 256),)](
            self.edge_offsets[-1:],
            self.center,
            self.neighbor,
            self.images,
            self.distances,
            self.short_flags,
            N=self.n_real,
            N_DUMMY=self.n_dummy,
            E_CAP=self.e_capacity,
            BLOCK=256,
            num_warps=4,
        )

    def _materialize_pairs(self) -> None:
        total_edges = self.edge_offsets[-1]
        total_clamped = total_edges.clamp(max=self.e_capacity)
        is_real = self.edge_ids < total_clamped
        center = self.center.long()
        neighbor = self.neighbor.long()
        images = self.images.long()
        self._image_bounds_valid.copy_(
            ((~is_real) | (images.abs() <= MAX_IMAGE).all(dim=1)).all().reshape(1)
        )
        radix = 2 * MAX_IMAGE + 1
        volume = radix**3
        forward_key = (
            (center * self.n_real + neighbor) * volume
            + (images[:, 0] + MAX_IMAGE) * radix * radix
            + (images[:, 1] + MAX_IMAGE) * radix
            + images[:, 2]
            + MAX_IMAGE
        )
        reverse_key = (
            (neighbor * self.n_real + center) * volume
            + (-images[:, 0] + MAX_IMAGE) * radix * radix
            + (-images[:, 1] + MAX_IMAGE) * radix
            - images[:, 2]
            + MAX_IMAGE
        )
        invalid_key = torch.iinfo(torch.int64).max - self.e_capacity + self.edge_ids
        forward_key = torch.where(is_real, forward_key, invalid_key)
        sorted_key, sort_idx = torch.sort(forward_key)
        self.center.copy_(self.center[sort_idx])
        self.neighbor.copy_(self.neighbor[sort_idx])
        self.images.copy_(self.images[sort_idx])
        self.distances.copy_(self.distances[sort_idx])
        self.short_flags.copy_(self.short_flags[sort_idx])

        center = self.center.long()
        neighbor = self.neighbor.long()
        images = self.images.long()
        reverse_key = (
            (neighbor * self.n_real + center) * volume
            + (-images[:, 0] + MAX_IMAGE) * radix * radix
            + (-images[:, 1] + MAX_IMAGE) * radix
            - images[:, 2]
            + MAX_IMAGE
        )
        reverse_key = torch.where(is_real, reverse_key, invalid_key)
        reverse_pos = torch.searchsorted(sorted_key, reverse_key).clamp(
            max=self.e_capacity - 1
        )
        pair_idx = reverse_pos
        reverse_matches = sorted_key[reverse_pos] == reverse_key
        canonical_row = torch.minimum(self.edge_ids, pair_idx)
        pair_leader = is_real & reverse_matches & (self.edge_ids <= pair_idx)
        pair_prefix = torch.cumsum(pair_leader.long(), dim=0)
        real_pair_id = pair_prefix[canonical_row] - 1
        real_u = total_clamped // 2
        pad_pair_id = real_u + (self.edge_ids - total_clamped).clamp_min(0) // 2
        d2u = torch.where(is_real, real_pair_id, pad_pair_id).clamp(
            min=0, max=self.u_capacity - 1
        )
        self.directed2undirected.copy_(d2u.int())

        sentinel = torch.full_like(self.edge_ids, self.u_capacity)
        scatter_index = torch.where(pair_leader, real_pair_id, sentinel).clamp(
            min=0, max=self.u_capacity
        )
        self._u2d_scatter.fill_(self.e_capacity)
        self._u2d_scatter.scatter_reduce_(
            0,
            scatter_index,
            self.edge_ids,
            reduce="amin",
            include_self=True,
        )
        pad_u2d = total_clamped + 2 * (self.u_ids - real_u).clamp_min(0)
        u2d = torch.where(self.u_ids < real_u, self._u2d_scatter[:-1], pad_u2d)
        self.undirected2directed.copy_(u2d.clamp(max=self.e_capacity - 1).int())

        pair_count = pair_leader.long().sum()
        invalid = (
            (total_edges.remainder(2) != 0)
            | (pair_count != total_edges // 2)
            | (is_real & ~reverse_matches).any()
        )
        self._invalid_pairs.copy_(invalid.reshape(1))

    def _materialize_bond_graph(self) -> None:
        total_edges = self.edge_offsets[-1].clamp(max=self.e_capacity)
        # Preserve the exact dist2 cutoff decision made by the materialization
        # kernel. Recomputing it from sqrt(dist2) can disagree at the boundary.
        is_short = (self.edge_ids < total_edges) & self.short_flags
        self.short_offsets[0].zero_()
        torch.cumsum(self.short_counts, dim=0, out=self.short_offsets[1:])
        short_global_rank = torch.cumsum(is_short.long(), dim=0) - 1
        safe_center = self.center.long().clamp(min=0, max=self.n_real - 1)
        short_local_rank = short_global_rank - self.short_offsets[safe_center]
        short_target = safe_center * self.buffer_max + short_local_rank
        short_target = torch.where(
            is_short,
            short_target,
            torch.full_like(short_target, self.n_real * self.buffer_max),
        ).clamp(min=0, max=self.n_real * self.buffer_max)
        self._short_scatter.fill_(self.e_capacity)
        self._short_scatter.scatter_reduce_(
            0,
            short_target,
            self.edge_ids,
            reduce="amin",
            include_self=True,
        )
        self.short_edge_matrix.copy_(
            self._short_scatter[:-1].reshape(self.n_real, self.buffer_max).int()
        )

        pair_counts = self.short_counts * (self.short_counts - 1)
        self.pair_offsets[0].zero_()
        torch.cumsum(pair_counts, dim=0, out=self.pair_offsets[1:])
        total_triplets = self.pair_offsets[-1]
        valid = self.triplet_ids < total_triplets.clamp(max=self.t_capacity)
        pair_atom = (
            torch.searchsorted(
                self.pair_offsets,
                self.triplet_ids,
                right=True,
            )
            - 1
        )
        pair_atom = pair_atom.clamp(min=0, max=self.n_real - 1)
        local_idx = self.triplet_ids - self.pair_offsets[pair_atom]
        group_size = self.short_counts[pair_atom]
        denom = (group_size - 1).clamp_min(1)
        first = local_idx // denom
        second_raw = local_idx % denom
        second = second_raw + (second_raw >= first).long()
        first_de = self.short_edge_matrix[
            pair_atom, first.clamp(max=self.buffer_max - 1)
        ]
        second_de = self.short_edge_matrix[
            pair_atom, second.clamp(max=self.buffer_max - 1)
        ]
        first_u = self.directed2undirected[
            first_de.long().clamp(min=0, max=self.e_capacity - 1)
        ]
        second_u = self.directed2undirected[
            second_de.long().clamp(min=0, max=self.e_capacity - 1)
        ]

        real_edges = total_edges
        real_u = real_edges // 2
        t_clamped = total_triplets.clamp(max=self.t_capacity)
        pad_id = (self.triplet_ids - t_clamped).clamp_min(0)
        n_pad_u = (self.u_capacity - real_u).clamp_min(1)
        pad_u = real_u + pad_id.remainder(n_pad_u)
        pad_de = real_edges + 2 * pad_id.remainder(n_pad_u)
        t_pad = (self.t_capacity - t_clamped).clamp_min(1)
        pad_atom = self.n_real + pad_id * self.n_dummy // t_pad

        self.bond_graph[:, 0].copy_(torch.where(valid, pair_atom, pad_atom).int())
        self.bond_graph[:, 1].copy_(torch.where(valid, first_u.long(), pad_u).int())
        self.bond_graph[:, 2].copy_(torch.where(valid, first_de.long(), pad_de).int())
        self.bond_graph[:, 3].copy_(torch.where(valid, second_u.long(), pad_u).int())
        self.bond_graph[:, 4].copy_(torch.where(valid, second_de.long(), pad_de).int())

    def _substitute_overflow_topology(self) -> None:
        overflow = self._overflow[0]
        self.center.copy_(torch.where(overflow, self._overflow_center, self.center))
        self.neighbor.copy_(
            torch.where(overflow, self._overflow_neighbor, self.neighbor)
        )
        self.images.copy_(torch.where(overflow, self._overflow_images, self.images))
        self.directed2undirected.copy_(
            torch.where(overflow, self._overflow_d2u, self.directed2undirected)
        )
        self.undirected2directed.copy_(
            torch.where(overflow, self._overflow_u2d, self.undirected2directed)
        )
        self.bond_graph.copy_(
            torch.where(overflow, self._overflow_bond_graph, self.bond_graph)
        )

    @torch.no_grad()
    def build(
        self,
        positions: Tensor,
        *,
        rebuild_candidates: bool = True,
    ) -> tuple[CrystalGraph, FixedCapacityGraphStatus]:
        """Build into persistent buffers and return graph plus device-side status."""
        with nvtx_range("fixed_capacity_graph_build"):
            positions = positions.to(device=self.device, dtype=torch.float32)
            if positions.shape != (self.n_real, 3):
                raise ValueError(
                    "positions must have shape "
                    f"{(self.n_real, 3)}, got {tuple(positions.shape)}"
                )
            frac_unwrapped = _right_matmul_3x3(positions, self.inv_cell)
            wraps = torch.where(
                self.pbc[0],
                torch.floor(frac_unwrapped),
                torch.zeros_like(frac_unwrapped),
            ).to(torch.int32)
            frac = frac_unwrapped - wraps.float()
            wrapped = _right_matmul_3x3(frac, self.cell)
            self.frac_coords[: self.n_real].copy_(frac)
            if rebuild_candidates:
                self.reference_positions.copy_(positions)
                self.reference_wraps.copy_(wraps)
                self.wrap_delta.zero_()
                self._build_candidates(wrapped)
                self._candidate_initialized = True
            elif not self._candidate_initialized:
                raise RuntimeError("refresh_candidates must run before candidate reuse")
            else:
                self.wrap_delta.copy_(wraps - self.reference_wraps)
            max_displacement = torch.linalg.vector_norm(
                positions - self.reference_positions,
                dim=1,
            ).max()
            candidate_valid = (
                max_displacement * 2.0 < self.skin
            ) & self._candidate_capacity_valid[0]
            self._max_displacement.copy_(max_displacement.reshape(1))
            self._candidate_valid.copy_(candidate_valid.reshape(1))
            self._materialize_edges(wrapped)
            self._materialize_pairs()
            self._materialize_bond_graph()
            overflow = (
                (self.edge_offsets[-1] >= self.e_capacity)
                | (self.pair_offsets[-1] > self.t_capacity)
                | self._invalid_pairs[0]
                | ~self._image_bounds_valid[0]
                | ~self._candidate_valid[0]
            )
            self._overflow.copy_(overflow.reshape(1))
            self._substitute_overflow_topology()
            self.graph.atom_graph[:, 0].copy_(self.center)
            self.graph.atom_graph[:, 1].copy_(self.neighbor)
            self.graph.neighbor_image.copy_(self.images)
            return self.graph, FixedCapacityGraphStatus(
                edge_count=self.edge_offsets[-1:],
                undirected_count=self.edge_offsets[-1:] // 2,
                triplet_count=self.pair_offsets[-1:],
                overflow=self._overflow,
                invalid_pairs=self._invalid_pairs,
                image_bounds_valid=self._image_bounds_valid,
                candidate_valid=self._candidate_valid,
                max_displacement=self._max_displacement,
                isolated_atom_count=self._isolated_atom_count,
            )
