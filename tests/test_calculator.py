import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.md.verlet import VelocityVerlet
from ase import units

from fastmd import FastMDCalculator, CUDAGraphConfig, ModelBackend, ModelCapabilities, register_model


class HarmonicModel(ModelBackend):
    capabilities = ModelCapabilities(frozenset({"energy", "forces", "stress"}))

    def __init__(self, **kwargs):
        super().__init__(device="cpu", cuda_graph=False, **kwargs)
        self.clears = 0

    def clear_graphs(self):
        self.clears += 1

    def _predict_eager(self, atoms, properties):
        result = {"energy": np.sum(atoms.positions ** 2) / 2, "forces": -atoms.positions.copy()}
        if "stress" in properties:
            result["stress"] = np.diag([1., 2., 3.])
        return result


def make_atoms(model=None):
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = FastMDCalculator(model or HarmonicModel())
    return atoms


def test_ase_results_cache_stress_and_free_energy():
    atoms = make_atoms()
    assert atoms.get_potential_energy(force_consistent=True) == atoms.get_potential_energy()
    np.testing.assert_allclose(atoms.get_forces(), -atoms.positions)
    assert atoms.calc.stats()["calls"] == 1
    np.testing.assert_array_equal(atoms.get_stress(), [1, 2, 3, 0, 0, 0])
    assert atoms.calc.stats()["calls"] == 2
    atoms.positions[0, 0] += .1
    atoms.get_forces()
    assert atoms.calc.stats()["calls"] == 3
    assert atoms.calc.stats()["invalidations"] == 0


@pytest.mark.parametrize("change", ["species", "order", "cell", "count"])
def test_capture_invalidated_for_structural_changes(change):
    atoms = make_atoms()
    atoms.numbers[0] = 6
    atoms.get_forces()
    backend = atoms.calc.backend
    previous = backend.clears
    if change == "species":
        atoms.numbers[0] = 8
    elif change == "order":
        atoms.numbers[:] = atoms.numbers[::-1]
    elif change == "cell":
        atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
    else:
        atoms += atoms[:1]
    atoms.get_forces()
    assert backend.clears == previous + 1


def test_warmup_does_not_move_atoms():
    atoms = make_atoms()
    positions, cell = atoms.positions.copy(), atoms.cell.array.copy()
    atoms.calc.warmup(atoms)
    np.testing.assert_array_equal(atoms.positions, positions)
    np.testing.assert_array_equal(atoms.cell.array, cell)
    assert atoms.calc.stats()["calls"] == 1
    atoms.get_forces()
    assert atoms.calc.stats()["calls"] == 1


def test_ase_constraints_and_md():
    atoms = make_atoms()
    atoms.set_constraint(FixAtoms(indices=[0]))
    before = atoms.positions.copy()
    VelocityVerlet(atoms, timestep=.1 * units.fs).run(3)
    np.testing.assert_array_equal(atoms.positions[0], before[0])
    assert not np.array_equal(atoms.positions[1], before[1])


def test_failed_prediction_leaves_no_stale_results():
    atoms = make_atoms()
    atoms.get_forces()
    atoms.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        atoms.get_forces()
    assert atoms.calc.results == {}


def test_partial_pbc_is_explicitly_rejected():
    atoms = make_atoms()
    atoms.pbc[2] = False
    with pytest.raises(ValueError, match="periodic"):
        atoms.get_forces()


def test_strict_cuda_does_not_silently_fallback():
    backend = HarmonicModel()
    backend.config = CUDAGraphConfig(enabled=True)
    with pytest.raises(RuntimeError, match="required"):
        make_atoms(backend).get_forces()


def test_auto_fallback_is_visible_once():
    backend = HarmonicModel()
    backend.config = CUDAGraphConfig()
    atoms = make_atoms(backend)
    with pytest.warns(RuntimeWarning, match="eager inference"):
        atoms.get_forces()
    assert atoms.calc.stats()["mode"] == "eager"
    assert "CUDA" in atoms.calc.stats()["reason"]


def test_results_are_owned():
    atoms = make_atoms()
    before = atoms.get_forces()
    saved = before.copy()
    atoms.positions += .2
    atoms.get_forces()
    np.testing.assert_array_equal(before, saved)


def test_custom_model_registry():
    def factory(checkpoint, device, cuda_graph):
        return HarmonicModel()
    register_model("test-harmonic", factory, overwrite=True)
    calc = FastMDCalculator("test-harmonic")
    assert isinstance(calc.backend, HarmonicModel)
    with pytest.raises(ValueError, match="already registered"):
        register_model("test-harmonic", factory)


def test_unknown_model_and_property():
    with pytest.raises(ValueError, match="available models"):
        FastMDCalculator("missing")
    with pytest.raises(PropertyNotImplementedError):
        make_atoms().get_magnetic_moments()


@pytest.mark.parametrize("setting", [{"enabled": "yes"}, {"enabled": 1}, {"warmup_steps": 0},
                                      {"edge_capacity_step": -1}, {"max_cached_graphs": 0},
                                      {"triplet_capacity_step": True}])
def test_bad_config(setting):
    with pytest.raises(ValueError):
        CUDAGraphConfig(**setting)
