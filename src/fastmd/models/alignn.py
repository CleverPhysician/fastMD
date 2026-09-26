"""ALIGNN-FF energy/force inference without an MD driver in the public API."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .base import ModelBackend, ModelCapabilities


class ALIGNNModel(ModelBackend):
    capabilities = ModelCapabilities(frozenset({"energy", "forces"}), frozenset({"energy", "forces"}))

    def __init__(self, *, checkpoint=None, intensive=True, force_multiplier=1.0,
                 force_mult_natoms=False, force_mult_batchsize=True, **kwargs):
        super().__init__(**kwargs)
        if checkpoint is None:
            raise ValueError("ALIGNN requires checkpoint='/path/to/model_directory' containing config.json and best_model.pt")
        path = Path(checkpoint).expanduser().resolve()
        config = json.loads((path / "config.json").read_text())
        if config["model"]["name"] != "alignn_atomwise":
            raise ValueError("fastMD currently supports ALIGNN-FF alignn_atomwise checkpoints only")
        # Stress is deliberately not advertised: original graph kernel provides EF.
        config["model"]["stresswise_weight"] = 0
        from fastmd._vendor.alignn.ff.calculators import AlignnAtomwiseCalculator
        self.calculator = AlignnAtomwiseCalculator(
            path=str(path), config=config, device=self.device, graph_device=self.device,
            include_stress=False, intensive=intensive, force_multiplier=force_multiplier,
            force_mult_natoms=force_mult_natoms, force_mult_batchsize=force_mult_batchsize,
        )
        self.model = self.calculator.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.runner = None
        self.builder = None
        self._captures = 0

    def graph_unavailable_reason(self, properties):
        reason = super().graph_unavailable_reason(properties)
        if reason:
            return reason
        from fastmd._vendor.alignn.gpu_graph_builder import gpu_graph_available
        if not gpu_graph_available(self.device):
            return "GPU neighbor operators missing; install fastMD[cuda]"
        config = self.calculator.config
        c = self.model.config
        if config["neighbor_strategy"] != "radius_graph" or c.alignn_layers <= 0:
            return "ALIGNN CUDA Graph requires radius_graph and alignn_layers > 0"
        if (c.include_pos_deriv or c.extra_features != 0 or c.atomwise_output_features > 0
                or c.additional_output_features > 0 or c.classification or c.use_cutoff_function
                or not c.calculate_gradient or not c.lg_on_fly or c.output_features != 1):
            return "This ALIGNN checkpoint configuration is not supported by CUDA Graph"
        return None

    def _predict_eager(self, atoms, properties):
        self.calculator.calculate(atoms.copy())
        return {key: self.calculator.results[key] for key in ("energy", "forces")}

    def _predict_graph(self, atoms, properties):
        from jarvis.core.specie import get_node_attributes
        from fastmd._vendor.alignn.gpu_graph_builder import TensorGraphBuilder
        from fastmd._vendor.alignn.ff.cuda_graph import ModelCUDAGraphRunner
        cfg = self.calculator.config
        if self.builder is None:
            features = np.asarray([get_node_attributes(species=symbol, atom_features=cfg["atom_features"])
                                   for symbol in atoms.get_chemical_symbols()])
            self.builder = TensorGraphBuilder(
                cell=torch.as_tensor(atoms.cell.array, dtype=torch.float32, device=self.device),
                pbc=torch.as_tensor(atoms.pbc, dtype=torch.bool, device=self.device),
                node_features=torch.as_tensor(features), cutoff=cfg["cutoff"],
                cutoff_extra=cfg.get("cutoff_extra", 0.5), device=self.device,
            )
        positions = torch.as_tensor(atoms.positions, dtype=torch.float32, device=self.device)
        tensors = self.builder.build_tensors(positions)
        n_edges, n_triplets = tensors[0].shape[0], tensors[3].shape[0]
        if self.runner is None or n_edges > self.runner.edge_capacity or n_triplets > self.runner.triplet_capacity:
            edge_step = self.config.edge_capacity_step or 1024
            triplet_step = self.config.triplet_capacity_step or 16384
            edge_capacity = ((n_edges + 512 + edge_step - 1) // edge_step) * edge_step
            triplet_capacity = ((n_triplets + 16384 + triplet_step - 1) // triplet_step) * triplet_step
            scale = self.calculator.force_multiplier
            if self.calculator.force_mult_natoms:
                scale *= len(atoms)
            if self.calculator.force_mult_batchsize:
                scale *= cfg["batch_size"]
            context = SimpleNamespace(model=self.model, graph_builder=self.builder, n_atoms=len(atoms),
                                      device=self.device, force_scale=scale, intensive=self.calculator.intensive,
                                      operator_fusions=self.config.enable_fusions)
            self.runner = ModelCUDAGraphRunner(context, tensors, edge_capacity=edge_capacity,
                                              triplet_capacity=triplet_capacity,
                                              capture_warmup=self.config.warmup_steps)
            self._captures += 1
        forces, energy = self.runner.run(tensors)
        return {"energy": float(energy.detach()), "forces": forces.detach().cpu().numpy().copy()}

    def clear_graphs(self):
        self.runner = self.builder = None

    def stats(self):
        return {**super().stats(), "captures": self._captures,
                "edge_capacity": self.runner.edge_capacity if self.runner else 0,
                "triplet_capacity": self.runner.triplet_capacity if self.runner else 0}
