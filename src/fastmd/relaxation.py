"""Device-resident FIRE relaxation for fixed-cell structures.

The optimizer keeps coordinates, velocities, and FIRE state on the callback's
device.  ``run_gpu_fire`` observes convergence at a configurable interval;
``run_gpu_fire_device`` keeps even the convergence flags on-device until
``GPUFireDeviceResult.to_host()`` is called.  The energy/force callback must
accept an ``(n_atoms, 3)`` tensor and return a scalar energy plus forces on the
same device.
"""

from dataclasses import dataclass, replace
import math
from typing import Callable

import torch


EnergyForce = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class GPUFireConfig:
    """FIRE parameters matching the fixed-cell opt3 protocol."""

    fmax: float = 0.02
    steps: int = 500
    dt: float = 0.1
    dt_max: float = 1.0
    maxstep: float = 0.2
    n_min: int = 5
    f_inc: float = 1.1
    f_dec: float = 0.5
    alpha_start: float = 0.1
    alpha_decay: float = 0.99
    check_interval: int = 1
    device_only: bool = False

    def __post_init__(self):
        for name in ("fmax", "dt", "dt_max", "maxstep", "f_inc", "f_dec",
                     "alpha_start", "alpha_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (("steps", 1), ("n_min", 0), ("check_interval", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.device_only, bool):
            raise ValueError("device_only must be a boolean")
        if self.dt_max < self.dt or self.f_inc <= 1 or self.f_dec >= 1:
            raise ValueError("require dt_max >= dt, f_inc > 1 and f_dec < 1")
        if self.alpha_start > 1 or self.alpha_decay > 1:
            raise ValueError("alpha_start and alpha_decay must be <= 1")


@dataclass
class GPUFireResult:
    positions: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    converged: bool
    completed_steps: int
    max_force: float
    evaluations: int


@dataclass
class GPUFireDeviceResult:
    """Device-resident result; call :meth:`to_host` at an observation boundary."""

    positions: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    converged: torch.Tensor
    completed_steps: torch.Tensor
    max_force: torch.Tensor
    invalid: torch.Tensor
    evaluations: int

    def to_host(self) -> GPUFireResult:
        if bool(self.invalid.item()):
            raise FloatingPointError("non-finite positions, energy or forces")
        return GPUFireResult(
            self.positions,
            self.energy,
            self.forces,
            bool(self.converged.item()),
            int(self.completed_steps.item()),
            float(self.max_force.item()),
            self.evaluations,
        )


def run_gpu_fire(positions: torch.Tensor, energy_and_forces: EnergyForce,
                 config: GPUFireConfig = GPUFireConfig()) -> GPUFireResult:
    """Run fixed-cell FIRE and materialize its summary on the host."""

    return _run_gpu_fire(positions, energy_and_forces, config).to_host()


def run_gpu_fire_device(positions: torch.Tensor, energy_and_forces: EnergyForce,
                        config: GPUFireConfig = GPUFireConfig()) -> GPUFireDeviceResult:
    """Run a fixed step budget without host convergence checks inside the loop."""

    return _run_gpu_fire(
        positions,
        energy_and_forces,
        replace(config, device_only=True, check_interval=0),
    )


def _run_gpu_fire(positions, energy_and_forces, config) -> GPUFireDeviceResult:
    if (not isinstance(positions, torch.Tensor) or positions.ndim != 2
            or positions.shape[-1] != 3 or positions.shape[0] == 0):
        raise ValueError("positions must have nonempty shape (n_atoms, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("positions must use float32 or float64 floating-point dtype")

    x = positions.detach().clone()
    invalid_positions = ~torch.isfinite(x).all()
    if not config.device_only and bool(invalid_positions.item()):
        raise ValueError("positions must be finite")
    if config.device_only:
        x = torch.where(invalid_positions, torch.zeros_like(x), x)

    def evaluate():
        with torch.enable_grad():
            energy, force = energy_and_forces(x.detach().clone())
        if not isinstance(energy, torch.Tensor) or not isinstance(force, torch.Tensor):
            raise ValueError("callback must return energy and forces as tensors")
        if energy.ndim != 0 or force.shape != x.shape:
            raise ValueError("callback must return scalar energy and (n_atoms, 3) forces")
        if energy.device != x.device or force.device != x.device:
            raise ValueError("callback outputs must stay on the positions device")
        return energy.detach(), force.detach().to(dtype=x.dtype)

    v = torch.zeros_like(x)
    dt = torch.full((), config.dt, dtype=x.dtype, device=x.device)
    alpha = torch.full((), config.alpha_start, dtype=x.dtype, device=x.device)
    n_positive = torch.zeros((), device=x.device, dtype=torch.long)
    completed = torch.zeros_like(n_positive)
    energy, forces = evaluate()
    evaluations = 1
    invalid = invalid_positions | ~torch.isfinite(energy) | ~torch.isfinite(forces).all()
    active = (forces.square().sum(dim=1).amax() >= config.fmax ** 2) & ~invalid
    if not config.device_only and bool(invalid.item()):
        raise FloatingPointError("callback returned non-finite energy or forces")

    if config.device_only or bool(active.item()):
        for step in range(config.steps):
            with torch.no_grad():
                if step > 0:
                    positive = (v * forces).sum() > 0
                    grow = positive & (n_positive > config.n_min)
                    mixed = ((1 - alpha) * v + alpha * forces
                             * torch.linalg.vector_norm(v)
                             / torch.linalg.vector_norm(forces).clamp_min(1e-30))
                    v = torch.where(positive, mixed, torch.zeros_like(v))
                    dt = torch.where(
                        positive,
                        torch.where(grow, (dt * config.f_inc).clamp_max(config.dt_max), dt),
                        dt * config.f_dec,
                    )
                    alpha = torch.where(
                        positive,
                        torch.where(grow, alpha * config.alpha_decay, alpha),
                        torch.full_like(alpha, config.alpha_start),
                    )
                    n_positive = torch.where(positive, n_positive + 1, torch.zeros_like(n_positive))
                v = v + dt * forces
                displacement = dt * v
                scale = (config.maxstep / torch.linalg.vector_norm(displacement).clamp_min(1e-30)).clamp_max(1)
                x = torch.where(active, x + displacement * scale, x)
                completed = completed + active.to(torch.long)

            candidate_energy, candidate_forces = evaluate()
            evaluations += 1
            energy = torch.where(active, candidate_energy, energy)
            forces = torch.where(active, candidate_forces, forces)
            invalid = invalid | ~torch.isfinite(energy) | ~torch.isfinite(forces).all()
            active = active & (forces.square().sum(dim=1).amax() >= config.fmax ** 2) & ~invalid
            if (not config.device_only and config.check_interval
                    and ((step + 1) % config.check_interval == 0 or step + 1 == config.steps)
                    and not bool(active.item())):
                break

    max_force = forces.square().sum(dim=1).amax().sqrt()
    return GPUFireDeviceResult(
        x.detach(),
        energy,
        forces,
        (max_force < config.fmax) & ~invalid,
        completed,
        max_force,
        invalid,
        evaluations,
    )


class _PersistentFIRE:
    """Mutable FIRE state whose tensor addresses remain stable during capture."""

    def __init__(self, positions, evaluate, config):
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
            raise ValueError("positions must have nonempty shape (n_atoms, 3)")
        if positions.dtype != torch.float64:
            raise ValueError("graph FIRE requires float64 optimizer positions")
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("positions must be finite")
        self.positions = positions.detach().clone()
        self.evaluate, self.config = evaluate, config
        self.velocity = torch.zeros_like(self.positions)
        self.dt = torch.full((), config.dt, dtype=positions.dtype, device=positions.device)
        self.alpha = torch.full_like(self.dt, config.alpha_start)
        self.n_positive = torch.zeros((), dtype=torch.long, device=positions.device)
        self.completed = torch.zeros_like(self.n_positive)
        energy, forces = self._evaluate()
        self.energy, self.forces = energy.clone(), forces.clone()
        self.invalid = ~torch.isfinite(energy) | ~torch.isfinite(forces).all()
        if bool(self.invalid):
            raise FloatingPointError("non-finite initial energy or forces")
        self.active = forces.square().sum(dim=1).amax() >= config.fmax ** 2

    def _evaluate(self):
        with torch.enable_grad():
            energy, forces = self.evaluate(self.positions.detach().clone())
        if (not isinstance(energy, torch.Tensor) or not isinstance(forces, torch.Tensor)
                or energy.ndim != 0 or forces.shape != self.positions.shape):
            raise ValueError("callback must return scalar energy and (n_atoms, 3) forces")
        if energy.device != self.positions.device or forces.device != self.positions.device:
            raise ValueError("energy and forces must stay on the positions device")
        return energy.detach(), forces.detach().to(self.positions.dtype)

    def tensors(self):
        return (self.positions, self.velocity, self.dt, self.alpha,
                self.n_positive, self.completed, self.energy, self.forces,
                self.invalid, self.active)

    def snapshot(self):
        return tuple(t.clone() for t in self.tensors())

    def restore_(self, snapshot):
        with torch.no_grad():
            for target, saved in zip(self.tensors(), snapshot, strict=True):
                target.copy_(saved)

    def step(self):
        self.advance()
        self.accept(*self._evaluate())

    def advance(self):
        c = self.config
        with torch.no_grad():
            velocity, forces, dt, alpha = self.velocity, self.forces, self.dt, self.alpha
            positive = (velocity * forces).sum() > 0
            grow = positive & (self.n_positive > c.n_min)
            mixed = ((1 - alpha) * velocity + alpha * forces
                     * torch.linalg.vector_norm(velocity)
                     / torch.linalg.vector_norm(forces).clamp_min(1e-30))
            later = self.completed > 0
            next_velocity = torch.where(
                later, torch.where(positive, mixed, torch.zeros_like(velocity)), velocity
            )
            next_dt = torch.where(
                later,
                torch.where(positive,
                            torch.where(grow, (dt * c.f_inc).clamp_max(c.dt_max), dt),
                            dt * c.f_dec),
                dt,
            )
            next_alpha = torch.where(
                later,
                torch.where(positive,
                            torch.where(grow, alpha * c.alpha_decay, alpha),
                            torch.full_like(alpha, c.alpha_start)),
                alpha,
            )
            next_positive = torch.where(
                later, torch.where(positive, self.n_positive + 1, torch.zeros_like(self.n_positive)),
                self.n_positive,
            )
            next_velocity = next_velocity + next_dt * forces
            displacement = next_dt * next_velocity
            scale = (c.maxstep / torch.linalg.vector_norm(displacement).clamp_min(1e-30)).clamp_max(1)
            self.positions.copy_(torch.where(
                self.active, self.positions + displacement * scale, self.positions
            ))
            self.velocity.copy_(torch.where(self.active, next_velocity, velocity))
            self.dt.copy_(torch.where(self.active, next_dt, dt))
            self.alpha.copy_(torch.where(self.active, next_alpha, alpha))
            self.n_positive.copy_(torch.where(self.active, next_positive, self.n_positive))
            self.completed.add_(self.active.to(torch.long))

    def accept(self, energy, forces):
        with torch.no_grad():
            self.energy.copy_(torch.where(self.active, energy, self.energy))
            self.forces.copy_(torch.where(self.active, forces, self.forces))
            self.invalid.logical_or_(~torch.isfinite(self.energy) | ~torch.isfinite(self.forces).all())
            self.active.logical_and_(
                (self.forces.square().sum(dim=1).amax() >= self.config.fmax ** 2) & ~self.invalid
            )

    def result(self, evaluations):
        maximum = self.forces.square().sum(dim=1).amax().sqrt()
        return GPUFireDeviceResult(
            self.positions.detach().clone(), self.energy.clone(), self.forces.clone(),
            (maximum < self.config.fmax) & ~self.invalid, self.completed,
            maximum, self.invalid, evaluations,
        ).to_host()


def run_gpu_fire_graph(positions: torch.Tensor, energy_and_forces: EnergyForce,
                       config: GPUFireConfig = GPUFireConfig(), *, warmup_steps=3):
    """Capture and replay one complete fixed-cell FIRE step on CUDA.

    The callback must already use fixed-shape device tensors.  This function
    deliberately rejects CPU, dynamic-neighbour growth, and ``device_only``
    configurations; there is no silent eager fallback.  It returns the normal
    result plus timing/capture statistics so setup cost is not hidden.
    """

    if getattr(energy_and_forces, "neighbor_overflow", "reject") == "grow":
        raise ValueError("neighbor growth is not supported by whole-step capture")
    if positions.device.type != "cuda":
        raise ValueError("CUDA is required for whole-step FIRE capture")
    if config.device_only:
        raise ValueError("graph runner materializes status; device_only is not supported")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
        raise ValueError("warmup_steps must be a positive integer")

    state = _PersistentFIRE(positions, energy_and_forces, config)
    stats = {
        "scope": "whole_fire_step",
        "capture_count": 0,
        "warmup_seconds": 0.0,
        "capture_seconds": 0.0,
        "production_replays": 0,
        "replay_seconds": 0.0,
    }
    if not bool(state.active):
        return state.result(1), stats

    import time

    snapshot = state.snapshot()
    addresses = tuple(t.data_ptr() for t in state.tensors())
    current = torch.cuda.current_stream(positions.device)
    side = torch.cuda.Stream(device=positions.device)
    side.wait_stream(current)
    started = time.perf_counter()
    with torch.cuda.stream(side):
        for _ in range(warmup_steps):
            state.restore_(snapshot)
            state.step()
        state.restore_(snapshot)
    current.wait_stream(side)
    torch.cuda.synchronize(positions.device)
    stats["warmup_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=side), torch.enable_grad():
            state.step()
        torch.cuda.synchronize(positions.device)
    except Exception as exc:
        raise RuntimeError("whole FIRE CUDA Graph capture failed; no eager fallback") from exc
    stats["capture_seconds"] = time.perf_counter() - started
    stats["capture_count"] = 1
    state.restore_(snapshot)
    if addresses != tuple(t.data_ptr() for t in state.tensors()):
        raise RuntimeError("FIRE persistent addresses changed during capture")
    torch.cuda.synchronize(positions.device)

    started = time.perf_counter()
    for step in range(config.steps):
        graph.replay()
        stats["production_replays"] += 1
        if config.check_interval and ((step + 1) % config.check_interval == 0 or step + 1 == config.steps):
            if not bool(state.active.item()):
                break
    torch.cuda.synchronize(positions.device)
    stats["replay_seconds"] = time.perf_counter() - started
    return state.result(stats["production_replays"] + 1), stats


__all__ = [
    "GPUFireConfig",
    "GPUFireResult",
    "GPUFireDeviceResult",
    "run_gpu_fire",
    "run_gpu_fire_device",
    "run_gpu_fire_graph",
]
