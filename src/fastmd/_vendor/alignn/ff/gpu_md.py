"""GPU-resident fixed-cell molecular dynamics for ALIGNN-FF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from jarvis.core.specie import get_node_attributes
from torch import Tensor

from fastmd._vendor.alignn.ff.calculators import AlignnAtomwiseCalculator
from fastmd._vendor.alignn.ff.gpu_integrator import GPUIntegrator
from fastmd._vendor.alignn.ff.gpu_logging import AsyncGPUMDLogger
from fastmd._vendor.alignn.gpu_graph_builder import TensorGraphBuilder


@dataclass
class GPUMDState:
    """Tensor state for one fixed-cell MD system."""

    positions: Tensor
    momenta: Tensor
    forces: Tensor | None
    potential_energy: Tensor | None


class GPUResidentMolecularDynamics:
    """Run ALIGNN-FF MD with state, graph construction, and integration on GPU.

    This path supports unconstrained, fixed-cell NVE and Berendsen NVT for one
    system. ASE is used only for initialization and optional output snapshots.
    """

    def __init__(
        self,
        atoms: Atoms,
        *,
        model_path: str | Path,
        device: torch.device | str = "cuda",
        ensemble: str = "nvt",
        temperature: float = 300.0,
        timestep: float = 1.0,
        taut: float = 100.0,
        trajectory=None,
        logfile=None,
        loginterval: int = 10,
        append_trajectory: bool = False,
        logger_ring: int = 8,
    ) -> None:
        if len(atoms) == 0:
            raise ValueError("GPU-resident MD requires at least one atom")
        if atoms.constraints:
            raise ValueError(
                "GPU-resident MD does not support ASE constraints"
            )
        if not np.all(atoms.get_pbc()) or atoms.cell.volume <= 0:
            raise ValueError("GPU-resident MD requires a periodic fixed cell")

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("GPUResidentMolecularDynamics requires CUDA")
        if self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(self.device)

        self.atoms_template = atoms.copy()
        self.n_atoms = len(atoms)
        self.timestep = float(timestep)
        self.nsteps = 0
        self._logged_initial = False

        loader = AlignnAtomwiseCalculator(
            path=str(Path(model_path).expanduser().resolve()),
            device=self.device,
            graph_device=self.device,
            include_stress=False,
        )
        self.model = loader.model
        self.config = loader.config
        if self.config["neighbor_strategy"] != "radius_graph":
            raise ValueError(
                "GPU-resident MD currently requires neighbor_strategy="
                "'radius_graph'"
            )
        if self.config["model"]["alignn_layers"] <= 0:
            raise ValueError(
                "GPU-resident MD currently requires an ALIGNN line graph"
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.cell = torch.as_tensor(
            atoms.cell.array,
            dtype=torch.float32,
            device=self.device,
        )
        self.pbc = torch.as_tensor(
            atoms.pbc,
            dtype=torch.bool,
            device=self.device,
        )
        self.masses = torch.as_tensor(
            atoms.get_masses(),
            dtype=torch.float32,
            device=self.device,
        )
        node_features = np.asarray(
            [
                get_node_attributes(
                    species=symbol,
                    atom_features=self.config["atom_features"],
                )
                for symbol in atoms.get_chemical_symbols()
            ]
        )
        self.graph_builder = TensorGraphBuilder(
            cell=self.cell,
            pbc=self.pbc,
            node_features=torch.as_tensor(node_features),
            cutoff=self.config["cutoff"],
            cutoff_extra=self.config.get("cutoff_extra", 0.5),
            device=self.device,
        )

        self.intensive = bool(loader.intensive)
        self.force_scale = float(loader.force_multiplier)
        if loader.force_mult_natoms:
            self.force_scale *= self.n_atoms
        if loader.force_mult_batchsize:
            self.force_scale *= self.config["batch_size"]

        positions = torch.as_tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
        )
        momenta_array = (
            atoms.get_momenta()
            if atoms.has("momenta")
            else np.zeros((self.n_atoms, 3), dtype=np.float32)
        )
        momenta = torch.as_tensor(
            momenta_array,
            dtype=torch.float32,
            device=self.device,
        )
        self.state = GPUMDState(
            positions=positions,
            momenta=momenta,
            forces=None,
            potential_energy=None,
        )
        self.integrator = GPUIntegrator(
            self.masses,
            dt_fs=timestep,
            ensemble=ensemble,
            temperature=temperature,
            taut_fs=taut,
            fix_com=None,
            degrees_of_freedom=self.n_atoms * 3,
        )
        self.logger = AsyncGPUMDLogger(
            atoms_template=self.atoms_template,
            device=self.device,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
            ring=logger_ring,
        )

    def build_graph(self, positions: Tensor):
        """Build atom and line graphs directly from CUDA positions."""
        return self.graph_builder.build(positions)

    def evaluate(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate force and extensive potential energy on the GPU."""
        graph, line_graph = self.build_graph(positions)
        with torch.enable_grad():
            result = self.model((graph, line_graph, self.cell))
        forces = result["grad"].detach() * self.force_scale
        energy = result["out"].reshape(-1)[0].detach()
        if self.intensive:
            energy = energy * self.n_atoms
        return forces, energy

    def force_fn(self, positions: Tensor) -> Tensor:
        """Evaluate force and cache the corresponding potential energy."""
        forces, energy = self.evaluate(positions)
        self.state.potential_energy = energy
        return forces

    def prepare(self) -> GPUMDState:
        """Evaluate the initial force without advancing the trajectory."""
        if self.state.forces is None:
            self.state.forces = self.force_fn(self.state.positions)
        return self.state

    def _log_current(self) -> None:
        if not self.logger.enabled:
            return
        self.prepare()
        assert self.state.forces is not None
        assert self.state.potential_energy is not None
        self.logger.enqueue(
            step=self.nsteps,
            time_ps=self.nsteps * self.timestep / 1000.0,
            positions=self.state.positions,
            momenta=self.state.momenta,
            forces=self.state.forces,
            cell=self.cell,
            potential_energy=self.state.potential_energy,
            kinetic_energy=self.integrator.kinetic_energy(self.state.momenta),
            temperature=self.integrator.temperature(self.state.momenta),
        )

    def step(self) -> GPUMDState:
        """Advance the trajectory by one step."""
        if self.logger.enabled and not self._logged_initial:
            self._log_current()
            self._logged_initial = True
        self.prepare()
        positions, momenta, forces = self.integrator.step(
            self.state.positions,
            self.state.momenta,
            self.state.forces,
            self.force_fn,
        )
        self.state = GPUMDState(
            positions=positions,
            momenta=momenta,
            forces=forces,
            potential_energy=self.state.potential_energy,
        )
        self.nsteps += 1
        if self.logger.should_log(self.nsteps):
            self._log_current()
        return self.state

    def run(self, steps: int) -> GPUMDState:
        """Run a requested number of MD steps."""
        for _ in range(int(steps)):
            self.step()
        return self.state

    def temperature(self) -> Tensor:
        """Return the current instantaneous temperature."""
        return self.integrator.temperature(self.state.momenta)

    def snapshot_atoms(self) -> Atoms:
        """Copy the current tensor state into an ASE Atoms object."""
        atoms = self.atoms_template.copy()
        atoms.set_positions(self.state.positions.detach().cpu().numpy())
        atoms.set_momenta(self.state.momenta.detach().cpu().numpy())
        return atoms

    def close(self) -> None:
        """Flush and close trajectory/log output resources."""
        self.logger.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
