"""Contract tests for the five WBM model backends."""

import numpy as np
import pytest
import torch
from ase.build import bulk

from fastmd import FastMDCalculator
from fastmd.models import load_model


WBM_MODELS = ("dpa4", "nequip", "orbv3", "sevennet", "tace")


@pytest.mark.parametrize("name", WBM_MODELS)
def test_wbm_models_are_registered(name):
    backend = load_model(name, checkpoint="unused", device="cpu", cuda_graph=False)
    assert backend.model_name == name
    assert backend.capabilities.properties == frozenset({"energy", "forces"})


class _FakeEvaluator:
    def __call__(self, positions):
        return positions.square().sum().reshape(()), -positions


@pytest.mark.parametrize("name", WBM_MODELS)
def test_wbm_eager_adapter_returns_owned_ase_results(name):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return _FakeEvaluator()

    calc = FastMDCalculator(
        name,
        checkpoint="unused",
        device="cpu",
        cuda_graph=False,
        model_kwargs={"evaluator_factory": factory},
    )
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = calc
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces, -atoms.positions)
    assert calls and calls[0]["route"] == "eager"
    assert calls[0]["model_name"] == name
    assert calls[0]["device"] == torch.device("cpu")

    saved = forces.copy()
    atoms.positions[0, 0] += 0.01
    atoms.get_forces()
    np.testing.assert_array_equal(forces, saved)
