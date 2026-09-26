"""Adapters for the five WBM model families.

The adapters deliberately keep the model-specific packages optional.  fastMD
only owns the ASE boundary and the eager/model-graph selection; the released
WBM evaluators continue to own model loading, neighbour construction and
their numerical conventions.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import torch

from .base import ModelBackend, ModelCapabilities


_GRAPH_MODELS = frozenset({"dpa4", "nequip", "orbv3", "sevennet"})


def _call_constructor(constructor, *args, **kwargs):
    """Pass only supported keyword arguments to versioned WBM evaluators."""
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return constructor(*args, **kwargs)
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return constructor(*args, **kwargs)
    accepted = {name for name in signature.parameters if name != "self"}
    return constructor(*args, **{key: value for key, value in kwargs.items() if key in accepted})


def _configure_dpa4_environment():
    # DeepMD reads these switches during its first import.
    os.environ.setdefault("DP_INTERFACE_PREC", "high")
    for name in ("ACT", "COMPILE", "CUDA", "CUTE", "TRITON", "TF32", "AMP"):
        os.environ.setdefault(f"DP_{name}_INFER", "0")


def _make_external_evaluator(model_name, route, atoms, checkpoint, device, config):
    """Build one released WBM evaluator for a single ASE structure."""
    if checkpoint is None:
        raise ValueError(f"{model_name} requires checkpoint='/path/to/checkpoint'")
    checkpoint = str(Path(checkpoint).expanduser())
    common = {"device": device, "compute_stress": False}

    if model_name == "dpa4":
        _configure_dpa4_environment()
        if route == "eager":
            from deepmd.md_stages.dpa4.opt1 import DPA4EnergyForceEvaluator as evaluator_type
        else:
            from deepmd.md_stages.dpa4.opt2 import DPA4ModelOnlyGraphEvaluator as evaluator_type
        return _call_constructor(evaluator_type, atoms, checkpoint, **common)

    if model_name == "nequip":
        if route == "eager":
            from nequip.md_stages.opt1 import EagerNequIPTorchSimEvaluator as evaluator_type
        else:
            from nequip.md_stages.opt2 import ModelOnlyCUDAGraphEvaluator as evaluator_type
        options = {
            "capture_energy_rtol": 2e-5,
            "capture_energy_atol": 2e-5,
            "capture_force_rtol": 2e-5,
            "capture_force_atol": 2e-4,
        }
        kwargs = {"require_stress": False, **common, "options": options}
        return _call_constructor(evaluator_type, atoms, checkpoint, **kwargs)

    if model_name == "orbv3":
        if route == "eager":
            from orb_models.md_stages.opt1 import OrbTorchSimEvaluator as evaluator_type
        else:
            from orb_models.md_stages.opt2 import ModelOnlyCUDAGraphEvaluator as evaluator_type
        kwargs = {
            "variant": "orb-v3-conservative-inf-mpa",
            "max_num_neighbors": None,
            **common,
            "capture_warmup": config.warmup_steps,
            "energy_atol": 2e-5,
            "force_atol": 2e-4,
        }
        evaluator = _call_constructor(evaluator_type, atoms, checkpoint, **kwargs)
        if route == "opt2" and callable(getattr(evaluator, "capture", None)):
            positions = torch.as_tensor(atoms.positions, dtype=torch.float64, device=device)
            evaluator.capture(positions)
        return evaluator

    if model_name == "sevennet":
        atomic_numbers = torch.as_tensor(atoms.numbers, dtype=torch.long, device=device)
        cell = torch.as_tensor(atoms.cell.array, dtype=torch.float64, device=device)
        pbc = torch.as_tensor(atoms.pbc, dtype=torch.bool, device=device)
        kwargs = {
            "atomic_numbers": atomic_numbers,
            "cell": cell,
            "pbc": pbc,
            "modal": None,
            **common,
            "capture_warmup": config.warmup_steps,
            "edge_margin": 0.25,
            "edge_step": 128,
        }
        if route == "eager":
            from sevenn.md_stages.opt1 import _SingleSystemPotential as evaluator_type
        else:
            from sevenn.md_stages.opt2 import _ModelOnlyCUDAGraphPotential as evaluator_type
        evaluator = _call_constructor(evaluator_type, checkpoint, **kwargs)
        if route == "opt2" and callable(getattr(evaluator, "capture", None)):
            positions = torch.as_tensor(atoms.positions, dtype=torch.float64, device=device)
            evaluator.capture(positions)
        return evaluator

    if model_name == "tace":
        # TACE's released fixed-neighbour session is a relaxation-level API;
        # expose its eager single-structure evaluator here and leave whole-step
        # capture to a future fastMD relaxation backend.
        from tace.md_stages.opt1 import TACETorchSimEvaluator
        return _call_constructor(TACETorchSimEvaluator, atoms, checkpoint, **common)

    raise ValueError(f"Unsupported WBM model: {model_name}")


def _normalise_device_output(model_name, output, positions):
    if isinstance(output, dict):
        energy, forces = output.get("energy"), output.get("forces")
    elif hasattr(output, "energy") and hasattr(output, "forces"):
        energy, forces = output.energy, output.forces
    elif isinstance(output, (tuple, list)) and len(output) == 3:
        forces, energy, _ = output
    elif isinstance(output, (tuple, list)) and len(output) == 2:
        first, second = output
        first_shape = getattr(first, "shape", ())
        if tuple(first_shape) == tuple(positions.shape):
            forces, energy = first, second
        else:
            energy, forces = first, second
    else:
        raise TypeError(f"{model_name} evaluator returned an unsupported output")
    if energy is None or forces is None:
        raise ValueError(f"{model_name} evaluator omitted energy or forces")
    energy = torch.as_tensor(energy)
    forces = torch.as_tensor(forces)
    if energy.device != positions.device or forces.device != positions.device:
        raise ValueError(f"{model_name} evaluator moved results off {positions.device}")
    if tuple(forces.shape) != tuple(positions.shape) or energy.numel() != 1:
        raise ValueError(f"{model_name} output shape does not match one structure")
    return energy.detach().reshape(()).to(torch.float64), forces.detach().to(torch.float64)


def _normalise_output(model_name, output, positions):
    energy, forces = _normalise_device_output(model_name, output, positions)
    return {
        "energy": energy.cpu().numpy().item(),
        "forces": forces.cpu().numpy().copy(),
    }


class WBMModelBackend(ModelBackend):
    """Common fastMD backend for DPA4, NequIP, ORB-v3, SevenNet and TACE."""

    capabilities = ModelCapabilities(
        frozenset({"energy", "forces"}),
        frozenset({"energy", "forces"}),
    )
    model_name = "wbm"

    def __init__(self, *, checkpoint=None, evaluator_factory=None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self._evaluator_factory = evaluator_factory or _make_external_evaluator
        self._evaluator = None
        self._route = (
            "opt2"
            if self.model_name in _GRAPH_MODELS
            and self.device.type == "cuda"
            and self.config.enabled is not False
            else "eager"
        )

    def _ensure_evaluator(self, atoms):
        if self._evaluator is None:
            self._evaluator = self._evaluator_factory(
                model_name=self.model_name,
                route=self._route,
                atoms=atoms.copy(),
                checkpoint=self.checkpoint,
                device=self.device,
                config=self.config,
            )
        return self._evaluator

    def _predict(self, atoms):
        positions = torch.as_tensor(atoms.positions, dtype=torch.float64, device=self.device)
        output = self._ensure_evaluator(atoms)(positions)
        return _normalise_output(self.model_name, output, positions)

    def device_callback(self, atoms):
        """Return the WBM evaluator without a per-step D2H conversion.

        The evaluator and its fixed model topology are created once for the
        structure.  The returned callback accepts device-resident float64
        coordinates and returns device-resident energy and forces for the
        GPU FIRE driver.
        """
        if self.device.type != "cuda":
            raise RuntimeError("device-native relaxation requires a CUDA backend")
        evaluator = self._ensure_evaluator(atoms)
        expected_shape = (len(atoms), 3)

        def evaluate(positions):
            if (positions.device != self.device or positions.dtype != torch.float64
                    or tuple(positions.shape) != expected_shape):
                raise ValueError("positions must match the WBM evaluator device, dtype and shape")
            return _normalise_device_output(self.model_name, evaluator(positions), positions)

        return evaluate

    def _predict_eager(self, atoms, properties):
        return self._predict(atoms)

    def _predict_graph(self, atoms, properties):
        return self._predict(atoms)

    def clear_graphs(self):
        self._evaluator = None

    def stats(self):
        return {**super().stats(), "model": self.model_name, "route": self._route}


class DPA4Model(WBMModelBackend):
    model_name = "dpa4"


class NequIPModel(WBMModelBackend):
    model_name = "nequip"


class ORBV3Model(WBMModelBackend):
    model_name = "orbv3"


class SevenNetModel(WBMModelBackend):
    model_name = "sevennet"


class TACEModel(WBMModelBackend):
    model_name = "tace"
    capabilities = ModelCapabilities(frozenset({"energy", "forces"}), frozenset())


__all__ = [
    "DPA4Model",
    "NequIPModel",
    "ORBV3Model",
    "SevenNetModel",
    "TACEModel",
    "WBMModelBackend",
]
