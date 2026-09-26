"""GPU-resident molecular-dynamics integrators."""

from __future__ import annotations

from collections.abc import Callable

import torch
from ase import units
from torch import Tensor


class GPUIntegrator:
    """GPU-resident Velocity-Verlet and Berendsen NVT integrator.

    The equations mirror ASE for unconstrained, fixed-cell systems. Positions,
    momenta, forces, and masses remain tensors throughout the MD hot loop.
    """

    def __init__(
        self,
        masses: Tensor,
        dt_fs: float,
        ensemble: str = "nvt",
        temperature: float = 300.0,
        taut_fs: float = 100.0,
        fix_com: bool | None = None,
        degrees_of_freedom: int | None = None,
    ) -> None:
        ensemble = ensemble.lower()
        if ensemble not in {"nve", "nvt"}:
            raise ValueError(f"unsupported GPUIntegrator ensemble: {ensemble}")
        self.masses = masses.reshape(-1, 1)
        self.dt = float(dt_fs) * units.fs
        self.ensemble = ensemble
        self.temperature_target = float(temperature)
        self.taut = float(taut_fs) * units.fs
        self.fix_com = (
            ensemble == "nvt" if fix_com is None else bool(fix_com)
        )
        self.degrees_of_freedom = (
            int(degrees_of_freedom)
            if degrees_of_freedom is not None
            else int(self.masses.numel() * 3)
        )

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        """Return kinetic energy for one system or a batch of systems."""
        kinetic = 0.5 * momenta * momenta / self.masses
        if momenta.ndim == 2:
            return kinetic.sum()
        if momenta.ndim == 3:
            return kinetic.sum(dim=(-2, -1))
        raise ValueError(
            "momenta must have shape [N, 3] or [B, N, 3], "
            f"got {tuple(momenta.shape)}"
        )

    def temperature(self, momenta: Tensor) -> Tensor:
        """Return instantaneous temperature in kelvin."""
        return 2.0 * self.kinetic_energy(momenta) / (
            self.degrees_of_freedom * units.kB
        )

    def scale_velocities(self, momenta: Tensor) -> Tensor:
        """Apply the ASE-compatible Berendsen thermostat scale factor."""
        if self.ensemble != "nvt":
            return momenta
        old_temperature = self.temperature(momenta).clamp_min(1e-12)
        scale = torch.sqrt(
            1.0
            + (self.temperature_target / old_temperature - 1.0)
            * (self.dt / self.taut)
        )
        scale = torch.clamp(scale, 0.9, 1.1)
        if momenta.ndim == 3:
            scale = scale[:, None, None]
        return momenta * scale

    def step(
        self,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None,
        force_fn: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance one Velocity-Verlet/Berendsen step on the tensor device."""
        momenta = self.scale_velocities(momenta)
        if forces is None:
            forces = force_fn(positions)

        momenta = momenta + 0.5 * self.dt * forces
        if self.fix_com:
            atom_dim = -2
            momenta = momenta - momenta.sum(
                dim=atom_dim,
                keepdim=True,
            ) / float(momenta.shape[atom_dim])

        positions = positions + self.dt * momenta / self.masses
        forces = force_fn(positions)
        momenta = momenta + 0.5 * self.dt * forces
        return positions, momenta, forces
