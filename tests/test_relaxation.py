import numpy as np
import pytest
import torch

from ase.build import bulk

from fastmd import FastMDCalculator, GPUFireConfig
from fastmd.relaxation import run_gpu_fire, run_gpu_fire_device, run_gpu_fire_graph
from fastmd.models import ModelBackend, ModelCapabilities


def harmonic_callback(positions):
    return positions.square().sum() / 2, -positions


def test_gpu_fire_converges_without_host_optimizer_state():
    positions = torch.tensor([[1.0, -0.5, 0.25], [-0.3, 0.4, -0.2]], dtype=torch.float64)
    result = run_gpu_fire(
        positions,
        harmonic_callback,
        GPUFireConfig(fmax=1e-3, steps=500, check_interval=10),
    )

    assert result.converged
    assert result.completed_steps <= 500
    # A positive check interval may perform a few frozen evaluations before
    # the next host-side convergence observation.
    assert result.evaluations >= result.completed_steps + 1
    assert result.max_force < 1e-3
    np.testing.assert_allclose(result.positions.numpy(), 0.0, atol=2e-3)


def test_gpu_fire_device_result_defers_status_materialization():
    positions = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float64)
    result = run_gpu_fire_device(
        positions,
        harmonic_callback,
        GPUFireConfig(fmax=1e-3, steps=40),
    )

    assert isinstance(result.converged, torch.Tensor)
    assert isinstance(result.completed_steps, torch.Tensor)
    host = result.to_host()
    assert host.converged
    assert host.max_force < 1e-3


def test_whole_step_graph_requires_cuda():
    positions = torch.zeros((1, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="CUDA"):
        run_gpu_fire_graph(positions, harmonic_callback)


class DeviceHarmonic(ModelBackend):
    capabilities = ModelCapabilities(frozenset({"energy", "forces"}), frozenset())

    def __init__(self):
        super().__init__(device="cpu", cuda_graph=False)

    def _predict_eager(self, atoms, properties):
        return {"energy": np.sum(atoms.positions ** 2) / 2, "forces": -atoms.positions.copy()}

    def device_callback(self, atoms):
        return harmonic_callback


def test_calculator_relax_gpu_updates_atoms_from_device_result():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = FastMDCalculator(DeviceHarmonic())
    before = atoms.positions.copy()

    result = atoms.calc.relax_gpu(
        atoms,
        GPUFireConfig(fmax=1e-3, steps=100, check_interval=5),
    )

    assert result.converged
    assert not np.array_equal(atoms.positions, before)
    np.testing.assert_allclose(atoms.positions, result.positions.numpy(), atol=0.0)
