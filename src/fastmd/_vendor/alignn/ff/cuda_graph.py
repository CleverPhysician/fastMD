"""Whole-step CUDA Graph replay for fixed-cell ALIGNN-FF molecular dynamics.

The implementation follows the MatRIS ``opt`` branch design: use a static
candidate-neighbor superset, filter the currently active edges on the device,
and capture force evaluation plus integration as one CUDA Graph. Candidate
coverage is checked after every replay by default, and a failed transaction is
rolled back before recapture.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import dgl
import torch
from torch import Tensor
from torch.autograd import grad
from torch.nn import functional as functional

from fastmd._vendor.alignn.ff.gpu_logging import AsyncGPUMDLogger
from fastmd._vendor.alignn.ff.gpu_md import GPUMDState, GPUResidentMolecularDynamics
from fastmd._vendor.alignn.graphs import compute_bond_cosines
from fastmd._vendor.alignn.gpu_graph_builder import neighbor_list_nvidia


_FUSED_LAYERNORM_MIN_ROWS = int(
    os.getenv("ALIGNN_FUSED_LAYERNORM_MIN_ROWS", "4096")
)
_FUSED_SEGMENT_MIN_ROWS = int(
    os.getenv("ALIGNN_FUSED_SEGMENT_MIN_ROWS", "4096")
)
_FUSED_ANGLE_RBF_MIN_ROWS = int(
    os.getenv("ALIGNN_FUSED_ANGLE_RBF_MIN_ROWS", "4096")
)
_FUSED_NODE_PROJECTIONS_MIN_ROWS = int(
    os.getenv("ALIGNN_FUSED_NODE_PROJECTIONS_MIN_ROWS", "4096")
)
_COMPACT_TRIPLET_STEP = int(
    os.getenv("ALIGNN_COMPACT_TRIPLET_STEP", "8192")
)
_COMPACT_TRIPLET_MARGIN = int(
    os.getenv("ALIGNN_COMPACT_TRIPLET_MARGIN", "8192")
)
_COMPACT_TRIPLET_SCALE = float(
    os.getenv("ALIGNN_COMPACT_TRIPLET_SCALE", "1.05")
)
_COMPACT_TRIPLET_INTERLEAVE = int(
    os.getenv("ALIGNN_COMPACT_TRIPLET_INTERLEAVE", "128")
)


class CandidateCoverageError(RuntimeError):
    """Raised when an exact neighbor is absent from the captured superset."""


class ModelGraphCapacityError(RuntimeError):
    """Raised when an exact graph exceeds a captured capacity bucket."""

    def __init__(self, edge_count: int, triplet_count: int) -> None:
        super().__init__(
            "CUDA Graph capacity exceeded: "
            f"edges={edge_count}, triplets={triplet_count}"
        )
        self.edge_count = int(edge_count)
        self.triplet_count = int(triplet_count)


@dataclass
class CUDAGraphStats:
    """Whole-step capture and replay counters."""

    captures: int = 0
    recaptures: int = 0
    launches: int = 0
    candidate_checks: int = 0
    coverage_failures: int = 0
    capacity_failures: int = 0
    max_active_edges: int = 0
    max_active_triplets: int = 0


def _edge_keys(
    center: Tensor,
    neighbor: Tensor,
    images: Tensor,
    n_atoms: int,
) -> Tensor:
    """Encode a periodic directed edge as a sortable int64 key."""
    max_image = 16
    radix = 2 * max_image + 1
    images = images.long()
    if images.numel() and bool((images.abs() > max_image).any().item()):
        raise RuntimeError("periodic image exceeds CUDA Graph key range")
    return (
        (center.long() * n_atoms + neighbor.long()) * radix**3
        + (images[:, 0] + max_image) * radix * radix
        + (images[:, 1] + max_image) * radix
        + images[:, 2]
        + max_image
    )


def _masked_edge_convolution(
    layer,
    source: Tensor,
    destination: Tensor,
    n_nodes: int,
    node_features: Tensor,
    edge_features: Tensor,
    edge_mask: Tensor,
    node_mask: Tensor | None = None,
    operator_fusions: bool = True,
) -> tuple[Tensor, Tensor]:
    """Evaluate ALIGNN's edge-gated convolution with tensor reductions."""
    mask = edge_mask.to(dtype=edge_features.dtype)
    use_fused_node_projections = (
        operator_fusions
        and node_features.is_cuda
        and node_features.shape[0] >= _FUSED_NODE_PROJECTIONS_MIN_ROWS
        and not any(
            parameter.requires_grad
            for projection in (
                layer.src_gate,
                layer.dst_gate,
                layer.src_update,
                layer.dst_update,
            )
            for parameter in projection.parameters()
        )
        and all(
            projection.bias is not None
            for projection in (
                layer.src_gate,
                layer.dst_gate,
                layer.src_update,
                layer.dst_update,
            )
        )
    )
    if use_fused_node_projections:
        projections = (
            layer.src_gate,
            layer.dst_gate,
            layer.src_update,
            layer.dst_update,
        )
        signature = tuple(
            (
                projection.weight.data_ptr(),
                projection.weight._version,
                projection.bias.data_ptr(),
                projection.bias._version,
            )
            for projection in projections
        )
        cached = getattr(layer, "_cuda_graph_node_projections", None)
        if cached is None or cached[0] != signature:
            cached = (
                signature,
                torch.cat(
                    tuple(projection.weight for projection in projections)
                ).contiguous(),
                torch.cat(
                    tuple(projection.bias for projection in projections)
                ).contiguous(),
                tuple(projection.out_features for projection in projections),
            )
            object.__setattr__(layer, "_cuda_graph_node_projections", cached)
        projected = functional.linear(node_features, cached[1], cached[2])
        (
            source_values,
            destination_values,
            source_update,
            transformed,
        ) = projected.split(cached[3], dim=1)
    else:
        source_values = layer.src_gate(node_features)
        destination_values = layer.dst_gate(node_features)
        source_update = layer.src_update(node_features)
        transformed = layer.dst_update(node_features)
    message = (
        source_values[source]
        + destination_values[destination]
        + layer.edge_gate(edge_features)
    )
    use_fused_segment = (
        operator_fusions
        and message.is_cuda
        and message.shape[0] >= _FUSED_SEGMENT_MIN_ROWS
    )
    if use_fused_segment:
        from fastmd._vendor.alignn.ff.ops.triton_gated_segment_sum import (
            gated_segment_sum,
        )

        sum_sigma_h, sum_sigma = gated_segment_sum(
            message,
            transformed,
            source,
            destination,
            mask,
            n_nodes,
        )
    else:
        sigma = torch.sigmoid(message)
        masked_sigma = sigma * mask
        sum_sigma_h = torch.zeros(
            (n_nodes, transformed.shape[-1]),
            dtype=transformed.dtype,
            device=transformed.device,
        ).index_add(0, destination, transformed[source] * masked_sigma)
        sum_sigma = torch.zeros_like(sum_sigma_h).index_add(
            0, destination, masked_sigma
        )
    nodes = source_update + sum_sigma_h / (sum_sigma + 1.0e-6)
    use_fused_node_norm = (
        operator_fusions
        and nodes.is_cuda
        and nodes.shape[0] >= _FUSED_LAYERNORM_MIN_ROWS
    )
    use_fused_edge_norm = (
        operator_fusions
        and message.is_cuda
        and message.shape[0] >= _FUSED_LAYERNORM_MIN_ROWS
    )
    if use_fused_node_norm or use_fused_edge_norm:
        from fastmd._vendor.alignn.ff.ops.triton_layernorm_silu import (
            fused_layernorm_silu,
        )

    if use_fused_node_norm:
        nodes = fused_layernorm_silu(
            nodes,
            layer.bn_nodes.weight,
            layer.bn_nodes.bias,
            layer.bn_nodes.eps,
            residual=node_features if layer.residual else None,
            row_mask=node_mask,
        )
    else:
        nodes = functional.silu(layer.bn_nodes(nodes))
    if use_fused_edge_norm:
        edges = fused_layernorm_silu(
            message,
            layer.bn_edges.weight,
            layer.bn_edges.bias,
            layer.bn_edges.eps,
            residual=edge_features if layer.residual else None,
            row_mask=mask,
        )
    else:
        edges = functional.silu(layer.bn_edges(message))
    if layer.residual:
        if not use_fused_node_norm:
            nodes = node_features + nodes
        if not use_fused_edge_norm:
            edges = edge_features + edges
    if node_mask is not None and not use_fused_node_norm:
        nodes = nodes * node_mask.to(dtype=nodes.dtype)
    if not use_fused_edge_norm:
        edges = edges * mask
    return nodes, edges


def _embedding_tail(
    embedding,
    basis: Tensor,
    *,
    operator_fusions: bool,
    final_mask: Tensor,
) -> Tensor:
    """Apply ALIGNN's two embedding MLPs after a precomputed RBF basis."""
    features = basis
    final_mask_fused = False
    mlp_layers = embedding[1:]
    for index, mlp_layer in enumerate(mlp_layers):
        linear = mlp_layer.layer[0]
        normalization = mlp_layer.layer[1]
        features = linear(features)
        use_fusion = (
            operator_fusions
            and features.is_cuda
            and features.shape[0] >= _FUSED_LAYERNORM_MIN_ROWS
        )
        is_last = index == len(mlp_layers) - 1
        if use_fusion:
            from fastmd._vendor.alignn.ff.ops.triton_layernorm_silu import (
                fused_layernorm_silu,
            )

            features = fused_layernorm_silu(
                features,
                normalization.weight,
                normalization.bias,
                normalization.eps,
                row_mask=final_mask if is_last else None,
            )
            final_mask_fused = is_last
        else:
            features = functional.silu(normalization(features))
    if not final_mask_fused:
        features = features * final_mask
    return features


def static_masked_alignn_forward(
    model,
    *,
    node_features: Tensor,
    source: Tensor,
    destination: Tensor,
    line_source: Tensor,
    line_destination: Tensor,
    displacement: Tensor,
    edge_mask: Tensor,
    triplet_mask: Tensor,
    operator_fusions: bool = True,
) -> tuple[Tensor, Tensor]:
    """Capture-safe single-graph ALIGNN energy and force evaluation."""
    config = model.config
    unsupported = (
        config.include_pos_deriv
        or config.extra_features != 0
        or config.atomwise_output_features > 0
        or config.additional_output_features > 0
        or config.stresswise_weight != 0
        or config.classification
        or config.use_cutoff_function
        or not config.calculate_gradient
        or config.output_features != 1
    )
    if unsupported:
        raise ValueError(
            "CUDA Graph MD does not support this ALIGNN model configuration"
        )

    atom_features = model.atom_embedding(node_features)
    displacement.requires_grad_(True)
    bond_length = torch.linalg.vector_norm(displacement, dim=1)
    bonds = model.edge_embedding(bond_length) * edge_mask
    use_fused_angle_rbf = (
        operator_fusions
        and displacement.is_cuda
        and line_source.shape[0] >= _FUSED_ANGLE_RBF_MIN_ROWS
    )
    if use_fused_angle_rbf:
        from fastmd._vendor.alignn.ff.ops.triton_angle_rbf import angle_rbf

        angle_expansion = model.angle_embedding[0]
        angle_basis = angle_rbf(
            displacement,
            line_source,
            line_destination,
            angle_expansion.centers,
            float(angle_expansion.gamma),
            triplet_mask,
        )
        angles = _embedding_tail(
            model.angle_embedding,
            angle_basis,
            operator_fusions=operator_fusions,
            final_mask=triplet_mask,
        )
    else:
        r1 = -displacement[line_source]
        r2 = displacement[line_destination]
        cosine = torch.sum(r1 * r2, dim=1) / (
            torch.linalg.vector_norm(r1, dim=1)
            * torch.linalg.vector_norm(r2, dim=1)
        )
        cosine = torch.clamp(cosine, -1.0, 1.0)
        angles = model.angle_embedding(cosine) * triplet_mask

    for layer in model.alignn_layers:
        atom_features, messages = _masked_edge_convolution(
            layer.node_update,
            source,
            destination,
            node_features.shape[0],
            atom_features,
            bonds,
            edge_mask,
            operator_fusions=operator_fusions,
        )
        bonds, angles = _masked_edge_convolution(
            layer.edge_update,
            line_source,
            line_destination,
            source.shape[0],
            messages,
            angles,
            triplet_mask,
            node_mask=edge_mask,
            operator_fusions=operator_fusions,
        )
    for layer in model.gcn_layers:
        atom_features, bonds = _masked_edge_convolution(
            layer,
            source,
            destination,
            node_features.shape[0],
            atom_features,
            bonds,
            edge_mask,
            operator_fusions=operator_fusions,
        )

    output = torch.squeeze(model.fc(atom_features.mean(dim=0, keepdim=True)))
    energy_for_gradient = output
    if config.energy_mult_natoms:
        energy_for_gradient = output * node_features.shape[0]
    if config.use_penalty:
        penalties = torch.where(
            bond_length < config.penalty_threshold,
            config.penalty_factor
            * (config.penalty_threshold - bond_length),
            torch.zeros_like(bond_length),
        )
        energy_for_gradient = energy_for_gradient + torch.sum(
            penalties * edge_mask.reshape(-1)
        )
    pair_forces = config.grad_multiplier * grad(
        energy_for_gradient,
        displacement,
        grad_outputs=torch.ones_like(energy_for_gradient),
        create_graph=True,
        retain_graph=True,
    )[0]
    if config.force_mult_natoms:
        pair_forces = pair_forces * node_features.shape[0]
    pair_forces = pair_forces * edge_mask
    forces = torch.zeros(
        (node_features.shape[0], pair_forces.shape[-1]),
        dtype=pair_forces.dtype,
        device=pair_forces.device,
    ).index_add(0, destination, pair_forces)
    if config.add_reverse_forces:
        forces = forces.index_add(0, source, -pair_forces)

    if config.link == "log":
        output = torch.exp(output)
    elif config.link == "logit":
        output = torch.sigmoid(output)
    return forces, output


class CandidateWholeStepGraphRunner:
    """Capture one or more complete ALIGNN-FF NVT steps per replay."""

    def __init__(
        self,
        dynamics: GPUResidentMolecularDynamics,
        *,
        candidate_skin: float = 0.5,
        steps_per_launch: int = 1,
        capture_warmup: int = 3,
        validate_interval: int = 1,
        operator_fusions: bool = True,
        minimum_triplet_capacity: int = 0,
    ) -> None:
        if candidate_skin <= 0:
            raise ValueError("candidate_skin must be positive")
        if steps_per_launch <= 0:
            raise ValueError("steps_per_launch must be positive")
        if capture_warmup < 1:
            raise ValueError("capture_warmup must be positive")
        if validate_interval < 1:
            raise ValueError("validate_interval must be positive")

        assert dynamics.state.forces is not None
        assert dynamics.state.potential_energy is not None
        self.model = dynamics.model
        self.integrator = dynamics.integrator
        self.graph_builder = dynamics.graph_builder
        self.device = dynamics.device
        self.cell = dynamics.cell
        self.inverse_cell = torch.linalg.inv(self.cell)
        self.pbc = dynamics.pbc
        self.n_atoms = dynamics.n_atoms
        self.cutoff = float(dynamics.config["cutoff"])
        self.candidate_skin = float(candidate_skin)
        self.steps_per_launch = int(steps_per_launch)
        self.capture_warmup = int(capture_warmup)
        self.validate_interval = int(validate_interval)
        self.operator_fusions = bool(operator_fusions)
        self.minimum_triplet_capacity = int(minimum_triplet_capacity)
        self.force_scale = dynamics.force_scale
        self.intensive = dynamics.intensive

        self.positions = dynamics.state.positions.clone()
        self.momenta = dynamics.state.momenta.clone()
        self.forces = dynamics.state.forces.clone()
        self.potential_energy = (
            dynamics.state.potential_energy.reshape(()).clone()
        )
        self.backup_positions = torch.empty_like(self.positions)
        self.backup_momenta = torch.empty_like(self.momenta)
        self.backup_forces = torch.empty_like(self.forces)
        self.backup_energy = torch.empty_like(self.potential_energy)
        self.active_edges = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        self.active_triplets = torch.zeros_like(self.active_edges)
        self._replays = 0

        self._build_candidate_graph(self.positions)
        self._prepare_triplet_compaction(self.positions)
        self._capture()

    def _wrapped(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        frac_unwrapped = positions @ self.inverse_cell
        wraps = torch.floor(frac_unwrapped).to(torch.int32)
        frac = frac_unwrapped - wraps.float()
        return frac @ self.cell, wraps, frac

    @torch.no_grad()
    def _neighbor_list(
        self,
        positions: Tensor,
        cutoff: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        wrapped, _, _ = self._wrapped(positions)
        batch_idx = torch.zeros(
            self.n_atoms,
            dtype=torch.int32,
            device=self.device,
        )
        center, neighbor, images, _ = neighbor_list_nvidia(
            wrapped,
            self.cell.unsqueeze(0),
            cutoff,
            batch_idx,
            self.pbc.unsqueeze(0),
            device=self.device,
        )
        return center, neighbor, images

    @torch.no_grad()
    def _build_candidate_graph(self, positions: Tensor) -> None:
        wrapped, wraps, frac = self._wrapped(positions)
        center, neighbor, images = self._neighbor_list(
            positions,
            self.cutoff + self.candidate_skin,
        )
        self.center = center
        self.neighbor = neighbor
        self.base_images = images
        self.reference_wraps = wraps.clone()

        graph = dgl.graph(
            (self.center, self.neighbor),
            num_nodes=self.n_atoms,
            device=self.device,
        )
        graph.ndata["atom_features"] = self.graph_builder.node_features
        graph.ndata["V"] = self.graph_builder.volume.expand(self.n_atoms)
        graph.ndata["frac_coords"] = frac
        displacement = wrapped[neighbor] - wrapped[center]
        displacement = displacement + images.float() @ self.cell
        graph.edata["r"] = displacement
        graph.edata["images"] = images.float()
        line_graph = graph.line_graph(shared=True)
        line_graph.apply_edges(compute_bond_cosines)
        line_source, line_destination = line_graph.edges()
        self.graph = graph
        self.line_graph = line_graph
        self.line_source = line_source
        self.line_destination = line_destination

    def _current_geometry(
        self,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        wrapped, wraps, frac = self._wrapped(positions)
        wrap_delta = wraps - self.reference_wraps
        images = self.base_images + (
            wrap_delta[self.neighbor] - wrap_delta[self.center]
        )
        displacement = wrapped[self.neighbor] - wrapped[self.center]
        displacement = displacement + images.float() @ self.cell
        edge_mask = torch.linalg.vector_norm(displacement, dim=1) < (
            self.cutoff + 1.0e-6
        )
        return displacement, images, edge_mask, frac

    @staticmethod
    def _round_capacity(required: int, step: int) -> int:
        return ((int(required) + int(step) - 1) // int(step)) * int(step)

    @torch.no_grad()
    def _prepare_triplet_compaction(self, positions: Tensor) -> None:
        """Allocate a padded active-triplet bucket for captured model work."""
        candidate_triplets = int(self.line_source.shape[0])
        self.triplet_capacity = candidate_triplets
        if self.operator_fusions and candidate_triplets:
            _, _, edge_mask, _ = self._current_geometry(positions)
            active_triplets = int(
                (
                    edge_mask[self.line_source]
                    & edge_mask[self.line_destination]
                )
                .sum()
                .item()
            )
            target = max(
                active_triplets + _COMPACT_TRIPLET_MARGIN,
                int(active_triplets * _COMPACT_TRIPLET_SCALE),
                self.minimum_triplet_capacity,
            )
            self.triplet_capacity = min(
                candidate_triplets,
                self._round_capacity(target, _COMPACT_TRIPLET_STEP),
            )
        # Keep padded rows distributed across real edge ids. Autograd's gather
        # backward still scatters zero gradients for masked rows, and filling
        # the tail with one repeated id creates severe atomic contention.
        self.compact_line_source = self.line_source[
            : self.triplet_capacity
        ].clone()
        self.compact_line_destination = self.line_destination[
            : self.triplet_capacity
        ].clone()
        compact_triplet_slots = torch.arange(
            self.triplet_capacity,
            dtype=torch.int64,
            device=self.device,
        )
        if self.triplet_capacity < candidate_triplets:
            self.triplet_interleave = min(
                max(1, _COMPACT_TRIPLET_INTERLEAVE),
                self.triplet_capacity,
            )
            while self.triplet_capacity % self.triplet_interleave:
                self.triplet_interleave -= 1
            stripe_size = (
                self.triplet_capacity // self.triplet_interleave
            )
            self.compact_triplet_ranks = (
                compact_triplet_slots % stripe_size
            ) * self.triplet_interleave + (
                compact_triplet_slots // stripe_size
            )
        else:
            self.triplet_interleave = 1
            self.compact_triplet_ranks = compact_triplet_slots

    def _evaluate(
        self,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        displacement, _, edge_mask, _ = self._current_geometry(
            positions
        )
        edge_mask_column = edge_mask[:, None].float()
        triplet_mask = (
            edge_mask[self.line_source]
            & edge_mask[self.line_destination]
        )
        triplet_count = triplet_mask.sum()
        model_line_source = self.line_source
        model_line_destination = self.line_destination
        triplet_mask_column = triplet_mask[:, None].float()
        if self.triplet_capacity < self.line_source.shape[0]:
            from fastmd._vendor.alignn.ff.ops.triton_compact_triplets import (
                compact_active_triplets,
            )

            triplet_count = compact_active_triplets(
                self.line_source,
                self.line_destination,
                triplet_mask,
                self.compact_line_source,
                self.compact_line_destination,
                self.triplet_interleave,
            )
            model_line_source = self.compact_line_source
            model_line_destination = self.compact_line_destination
            triplet_mask_column = (
                self.compact_triplet_ranks < triplet_count
            )[:, None].float()
        with torch.enable_grad():
            forces, energy = static_masked_alignn_forward(
                self.model,
                node_features=self.graph_builder.node_features,
                source=self.center,
                destination=self.neighbor,
                line_source=model_line_source,
                line_destination=model_line_destination,
                displacement=displacement,
                edge_mask=edge_mask_column,
                triplet_mask=triplet_mask_column,
                operator_fusions=self.operator_fusions,
            )
        forces = forces * self.force_scale
        if self.intensive:
            energy = energy * self.n_atoms
        return forces, energy, edge_mask.sum(), triplet_count

    def _step_body(self) -> None:
        with torch.no_grad():
            momenta = self.integrator.scale_velocities(self.momenta)
            momenta = momenta + 0.5 * self.integrator.dt * self.forces
            if self.integrator.fix_com:
                momenta = momenta - momenta.sum(
                    dim=0, keepdim=True
                ) / float(self.n_atoms)
            positions = (
                self.positions
                + self.integrator.dt * momenta / self.integrator.masses
            )
        forces, energy, edge_count, triplet_count = self._evaluate(positions)
        with torch.no_grad():
            momenta = momenta + 0.5 * self.integrator.dt * forces
            self.positions.copy_(positions)
            self.momenta.copy_(momenta)
            self.forces.copy_(forces)
            self.potential_energy.copy_(energy)
            self.active_edges.copy_(edge_count)
            self.active_triplets.copy_(triplet_count)

    def _window_body(self) -> None:
        self.backup_positions.copy_(self.positions)
        self.backup_momenta.copy_(self.momenta)
        self.backup_forces.copy_(self.forces)
        self.backup_energy.copy_(self.potential_energy)
        for _ in range(self.steps_per_launch):
            self._step_body()

    @torch.no_grad()
    def _restore_backup(self) -> None:
        self.positions.copy_(self.backup_positions)
        self.momenta.copy_(self.backup_momenta)
        self.forces.copy_(self.backup_forces)
        self.potential_energy.copy_(self.backup_energy)

    def _capture(self) -> None:
        saved = (
            self.positions.clone(),
            self.momenta.clone(),
            self.forces.clone(),
            self.potential_energy.clone(),
        )
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(self.capture_warmup):
                self._window_body()
                self.positions.copy_(saved[0])
                self.momenta.copy_(saved[1])
                self.forces.copy_(saved[2])
                self.potential_energy.copy_(saved[3])
        torch.cuda.current_stream(self.device).wait_stream(side)
        self.positions.copy_(saved[0])
        self.momenta.copy_(saved[1])
        self.forces.copy_(saved[2])
        self.potential_energy.copy_(saved[3])
        torch.cuda.synchronize(self.device)
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self._window_body()
        self.positions.copy_(saved[0])
        self.momenta.copy_(saved[1])
        self.forces.copy_(saved[2])
        self.potential_energy.copy_(saved[3])
        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def validate_candidate_coverage(self) -> None:
        """Verify every exact-cutoff edge is present in the candidate graph."""
        exact_center, exact_neighbor, exact_images = self._neighbor_list(
            self.positions,
            self.cutoff,
        )
        _, wraps, _ = self._wrapped(self.positions)
        wrap_delta = wraps - self.reference_wraps
        candidate_images = self.base_images + (
            wrap_delta[self.neighbor] - wrap_delta[self.center]
        )
        candidate_keys = torch.sort(
            _edge_keys(
                self.center,
                self.neighbor,
                candidate_images,
                self.n_atoms,
            )
        ).values
        exact_keys = _edge_keys(
            exact_center,
            exact_neighbor,
            exact_images,
            self.n_atoms,
        )
        locations = torch.searchsorted(candidate_keys, exact_keys)
        safe_locations = locations.clamp(max=candidate_keys.numel() - 1)
        missing = (locations >= candidate_keys.numel()) | (
            candidate_keys[safe_locations] != exact_keys
        )
        if bool(missing.any().item()):
            self._restore_backup()
            torch.cuda.synchronize(self.device)
            raise CandidateCoverageError(
                f"candidate graph missed {int(missing.sum().item())} edges"
            )

    def replay(self) -> GPUMDState:
        """Replay a captured window and validate its candidate coverage."""
        self.cuda_graph.replay()
        self._replays += 1
        active_triplets = int(self.active_triplets.item())
        if active_triplets > self.triplet_capacity:
            active_edges = int(self.active_edges.item())
            self._restore_backup()
            torch.cuda.synchronize(self.device)
            raise ModelGraphCapacityError(active_edges, active_triplets)
        if self._replays % self.validate_interval == 0:
            self.validate_candidate_coverage()
        return self.state

    @property
    def state(self) -> GPUMDState:
        return GPUMDState(
            positions=self.positions,
            momenta=self.momenta,
            forces=self.forces,
            potential_energy=self.potential_energy,
        )

    @torch.no_grad()
    def reset(self, state: GPUMDState) -> None:
        """Restore state without changing addresses captured by CUDA Graph."""
        assert state.forces is not None
        assert state.potential_energy is not None
        self.positions.copy_(state.positions)
        self.momenta.copy_(state.momenta)
        self.forces.copy_(state.forces)
        self.potential_energy.copy_(state.potential_energy.reshape(()))
        self._replays = 0


class CUDAGraphMolecularDynamics(GPUResidentMolecularDynamics):
    """ALIGNN-FF MD whose force evaluation and integrator use CUDA Graph."""

    def __init__(
        self,
        *args,
        candidate_skin: float = 0.5,
        steps_per_launch: int = 1,
        capture_warmup: int = 3,
        validate_interval: int = 1,
        operator_fusions: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.candidate_skin = float(candidate_skin)
        self.steps_per_launch = int(steps_per_launch)
        self.capture_warmup = int(capture_warmup)
        self.validate_interval = int(validate_interval)
        self.operator_fusions = bool(operator_fusions)
        self.runner: CandidateWholeStepGraphRunner | None = None
        self.cuda_stats = CUDAGraphStats()

    def _make_runner(self, minimum_triplet_capacity: int = 0) -> None:
        self.runner = CandidateWholeStepGraphRunner(
            self,
            candidate_skin=self.candidate_skin,
            steps_per_launch=self.steps_per_launch,
            capture_warmup=self.capture_warmup,
            validate_interval=self.validate_interval,
            operator_fusions=self.operator_fusions,
            minimum_triplet_capacity=minimum_triplet_capacity,
        )
        self.cuda_stats.captures += 1
        self.state = self.runner.state

    def prepare(self) -> GPUMDState:
        """Evaluate the initial force and capture the whole-step graph."""
        if self.runner is None:
            super().prepare()
            self._make_runner()
        return self.state

    def _recapture(self, minimum_triplet_capacity: int = 0) -> None:
        assert self.runner is not None
        restored = self.runner.state
        self.state = restored
        self.runner = None
        gc.collect()
        torch.cuda.empty_cache()
        self.cuda_stats.recaptures += 1
        self._make_runner(minimum_triplet_capacity)

    def step(self) -> GPUMDState:
        """Replay one captured window, recapturing after safe rollback."""
        if self.logger.enabled and not self._logged_initial:
            self._log_current()
            self._logged_initial = True
        self.prepare()
        assert self.runner is not None
        try:
            self.state = self.runner.replay()
        except CandidateCoverageError:
            self.cuda_stats.coverage_failures += 1
            self._recapture()
            assert self.runner is not None
            self.state = self.runner.replay()
        except ModelGraphCapacityError as error:
            self.cuda_stats.capacity_failures += 1
            self._recapture(
                error.triplet_count + _COMPACT_TRIPLET_MARGIN
            )
            assert self.runner is not None
            self.state = self.runner.replay()
        self.nsteps += self.steps_per_launch
        self.cuda_stats.launches += 1
        if self.nsteps % self.validate_interval == 0:
            self.cuda_stats.candidate_checks += 1
        self.cuda_stats.max_active_edges = max(
            self.cuda_stats.max_active_edges,
            int(self.runner.active_edges.item()),
        )
        self.cuda_stats.max_active_triplets = max(
            self.cuda_stats.max_active_triplets,
            int(self.runner.active_triplets.item()),
        )
        if self.logger.should_log(self.nsteps):
            self._log_current()
        return self.state

    def run(self, steps: int) -> GPUMDState:
        """Run a multiple of the captured steps-per-launch window."""
        steps = int(steps)
        if steps % self.steps_per_launch:
            raise ValueError("steps must be divisible by steps_per_launch")
        for _ in range(steps // self.steps_per_launch):
            self.step()
        return self.state

    def reset_state(self, state: GPUMDState) -> None:
        """Reset state while preserving captured tensor addresses."""
        self.prepare()
        assert self.runner is not None
        self.runner.reset(state)
        self.state = self.runner.state
        self.nsteps = 0
        self._logged_initial = False
        self.cuda_stats.launches = 0
        self.cuda_stats.candidate_checks = 0
        self.cuda_stats.max_active_edges = 0
        self.cuda_stats.max_active_triplets = 0

    def configure_logging(
        self,
        *,
        trajectory=None,
        logfile=None,
        loginterval: int = 10,
    ) -> None:
        """Enable asynchronous ASE output after warm-up/reset."""
        self.logger.close()
        self.logger = AsyncGPUMDLogger(
            atoms_template=self.atoms_template,
            device=self.device,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
        )

    def graph_stats(self) -> dict[str, int | float]:
        """Return capture, replay, and static topology statistics."""
        runner = self.runner
        return {
            **self.cuda_stats.__dict__,
            "steps_per_launch": self.steps_per_launch,
            "candidate_skin": self.candidate_skin,
            "operator_fusions": self.operator_fusions,
            "candidate_edges": (
                runner.graph.num_edges() if runner is not None else 0
            ),
            "candidate_triplets": (
                runner.line_graph.num_edges() if runner is not None else 0
            ),
            "model_triplet_capacity": (
                runner.triplet_capacity if runner is not None else 0
            ),
        }


class ModelCUDAGraphRunner:
    """Capture an exact, sink-masked ALIGNN forward in a capacity bucket."""

    def __init__(
        self,
        dynamics: GPUResidentMolecularDynamics,
        graph_tensors: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        *,
        edge_capacity: int,
        triplet_capacity: int,
        capture_warmup: int = 3,
    ) -> None:
        if capture_warmup < 1:
            raise ValueError("capture_warmup must be positive")
        self.model = dynamics.model
        self.node_features = dynamics.graph_builder.node_features
        self.n_atoms = dynamics.n_atoms
        self.device = dynamics.device
        self.force_scale = dynamics.force_scale
        self.intensive = dynamics.intensive
        self.edge_capacity = int(edge_capacity)
        self.triplet_capacity = int(triplet_capacity)
        self.capture_warmup = int(capture_warmup)
        self.operator_fusions = bool(dynamics.operator_fusions)

        self.source = torch.zeros(
            self.edge_capacity, dtype=torch.int64, device=self.device
        )
        self.destination = torch.zeros_like(self.source)
        self.displacement = torch.zeros(
            (self.edge_capacity, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.edge_mask = torch.zeros(
            (self.edge_capacity, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.line_source = torch.zeros(
            self.triplet_capacity,
            dtype=torch.int64,
            device=self.device,
        )
        self.line_destination = torch.zeros_like(self.line_source)
        self.triplet_mask = torch.zeros(
            (self.triplet_capacity, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.output_forces = torch.empty(
            (self.n_atoms, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.output_energy = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self.edge_count = 0
        self.triplet_count = 0
        self.update(graph_tensors)
        self._capture()

    @torch.no_grad()
    def update(
        self,
        graph_tensors: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> None:
        """Copy one exact ragged graph into persistent capacity buffers."""
        source, destination, displacement, line_source, line_destination = (
            graph_tensors
        )
        edge_count = source.shape[0]
        triplet_count = line_source.shape[0]
        if (
            edge_count > self.edge_capacity
            or triplet_count > self.triplet_capacity
        ):
            raise ModelGraphCapacityError(edge_count, triplet_count)
        self.source[:edge_count].copy_(source)
        self.destination[:edge_count].copy_(destination)
        self.displacement[:edge_count].copy_(displacement.detach())
        self.edge_mask.zero_()
        self.edge_mask[:edge_count].fill_(1.0)
        self.line_source[:triplet_count].copy_(line_source)
        self.line_destination[:triplet_count].copy_(line_destination)
        self.triplet_mask.zero_()
        self.triplet_mask[:triplet_count].fill_(1.0)
        self.edge_count = int(edge_count)
        self.triplet_count = int(triplet_count)

    def _body(self) -> None:
        forces, energy = static_masked_alignn_forward(
            self.model,
            node_features=self.node_features,
            source=self.source,
            destination=self.destination,
            line_source=self.line_source,
            line_destination=self.line_destination,
            displacement=self.displacement,
            edge_mask=self.edge_mask,
            triplet_mask=self.triplet_mask,
            operator_fusions=self.operator_fusions,
        )
        with torch.no_grad():
            self.output_forces.copy_(forces * self.force_scale)
            if self.intensive:
                energy = energy * self.n_atoms
            self.output_energy.copy_(energy)

    def _capture(self) -> None:
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(self.capture_warmup):
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self._body()
        torch.cuda.synchronize(self.device)

    def run(
        self,
        graph_tensors: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Update exact graph inputs and replay the captured model."""
        self.update(graph_tensors)
        self.cuda_graph.replay()
        return self.output_forces, self.output_energy


class ModelCUDAGraphMolecularDynamics(GPUResidentMolecularDynamics):
    """GPU-resident MD with exact dynamic graphs and captured ALIGNN forces."""

    def __init__(
        self,
        *args,
        edge_capacity_step: int = 1024,
        triplet_capacity_step: int = 16384,
        edge_capacity_margin: int = 512,
        triplet_capacity_margin: int = 16384,
        capture_warmup: int = 3,
        operator_fusions: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if edge_capacity_step <= 0 or triplet_capacity_step <= 0:
            raise ValueError("capacity steps must be positive")
        self.edge_capacity_step = int(edge_capacity_step)
        self.triplet_capacity_step = int(triplet_capacity_step)
        self.edge_capacity_margin = int(edge_capacity_margin)
        self.triplet_capacity_margin = int(triplet_capacity_margin)
        self.capture_warmup = int(capture_warmup)
        self.operator_fusions = bool(operator_fusions)
        self.model_runner: ModelCUDAGraphRunner | None = None
        self.model_graph_captures = 0
        self.model_graph_recaptures = 0
        self.model_graph_replays = 0
        self.max_edges = 0
        self.max_triplets = 0
        self.capacity_history: list[tuple[int, int]] = []

    @staticmethod
    def _round_capacity(required: int, step: int) -> int:
        return ((int(required) + int(step) - 1) // int(step)) * int(step)

    def _capacities(
        self,
        edge_count: int,
        triplet_count: int,
    ) -> tuple[int, int]:
        return (
            self._round_capacity(
                edge_count + self.edge_capacity_margin,
                self.edge_capacity_step,
            ),
            self._round_capacity(
                triplet_count + self.triplet_capacity_margin,
                self.triplet_capacity_step,
            ),
        )

    def _capture_model_runner(
        self,
        graph_tensors: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        *,
        minimum_edges: int | None = None,
        minimum_triplets: int | None = None,
    ) -> None:
        edge_count = max(graph_tensors[0].shape[0], minimum_edges or 0)
        triplet_count = max(graph_tensors[3].shape[0], minimum_triplets or 0)
        edge_capacity, triplet_capacity = self._capacities(
            edge_count, triplet_count
        )
        self.model_runner = ModelCUDAGraphRunner(
            self,
            graph_tensors,
            edge_capacity=edge_capacity,
            triplet_capacity=triplet_capacity,
            capture_warmup=self.capture_warmup,
        )
        self.model_graph_captures += 1
        self.capacity_history.append((edge_capacity, triplet_capacity))

    def prepare(self) -> GPUMDState:
        """Prepare the initial force and capture the first model bucket."""
        super().prepare()
        if self.model_runner is None:
            graph_tensors = self.graph_builder.build_tensors(
                self.state.positions
            )
            self._capture_model_runner(graph_tensors)
        return self.state

    def evaluate(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Build the exact graph eagerly and replay captured ALIGNN forces."""
        if self.model_runner is None:
            return super().evaluate(positions)
        graph_tensors = self.graph_builder.build_tensors(positions)
        edge_count = graph_tensors[0].shape[0]
        triplet_count = graph_tensors[3].shape[0]
        self.max_edges = max(self.max_edges, edge_count)
        self.max_triplets = max(self.max_triplets, triplet_count)
        try:
            forces, energy = self.model_runner.run(graph_tensors)
        except ModelGraphCapacityError as overflow:
            self.model_graph_recaptures += 1
            self.model_runner = None
            gc.collect()
            torch.cuda.empty_cache()
            self._capture_model_runner(
                graph_tensors,
                minimum_edges=overflow.edge_count,
                minimum_triplets=overflow.triplet_count,
            )
            assert self.model_runner is not None
            forces, energy = self.model_runner.run(graph_tensors)
        self.model_graph_replays += 1
        return forces, energy

    def reset_state(self, state: GPUMDState) -> None:
        """Restore an initial state while retaining captured model buffers."""
        assert state.forces is not None
        assert state.potential_energy is not None
        self.state = GPUMDState(
            positions=state.positions.clone(),
            momenta=state.momenta.clone(),
            forces=state.forces.clone(),
            potential_energy=state.potential_energy.clone(),
        )
        self.nsteps = 0
        self._logged_initial = False
        self.model_graph_replays = 0
        self.max_edges = 0
        self.max_triplets = 0

    def configure_logging(
        self,
        *,
        trajectory=None,
        logfile=None,
        loginterval: int = 10,
    ) -> None:
        """Enable asynchronous ASE output after warm-up/reset."""
        self.logger.close()
        self.logger = AsyncGPUMDLogger(
            atoms_template=self.atoms_template,
            device=self.device,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
        )

    def graph_stats(self) -> dict[str, object]:
        """Return capture capacities and production replay statistics."""
        runner = self.model_runner
        return {
            "captures": self.model_graph_captures,
            "recaptures": self.model_graph_recaptures,
            "replays": self.model_graph_replays,
            "edge_capacity": runner.edge_capacity if runner else 0,
            "triplet_capacity": runner.triplet_capacity if runner else 0,
            "max_edges": self.max_edges,
            "max_triplets": self.max_triplets,
            "capacity_history": list(self.capacity_history),
            "operator_fusions": self.operator_fusions,
        }
