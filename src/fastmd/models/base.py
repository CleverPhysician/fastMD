"""Small backend contract, independent of a particular graph representation."""
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import warnings

import numpy as np
import torch
from ase.calculators.calculator import PropertyNotImplementedError

from fastmd.config import CUDAGraphConfig


@dataclass(frozen=True)
class ModelCapabilities:
    properties: frozenset[str] = frozenset({"energy", "forces"})
    cuda_graph_properties: frozenset[str] = frozenset()
    periodic_only: bool = True


def resolve_device(device):
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("fastMD currently supports device='cpu' or 'cuda[:index]'")
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use device='cpu' or a CUDA PyTorch environment.")
        if result.index is None:
            result = torch.device("cuda", torch.cuda.current_device())
        if result.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {result}")
    return result


class ModelBackend(ABC):
    """Return owned NumPy results in ASE units, with total (extensive) energy.

    Implement _predict_eager; CUDA adapters also implement _predict_graph and
    clear_graphs. Graphs must never outlive a change in species/order/cell/PBC.
    One backend belongs to one calculator and must not be used concurrently.
    """

    capabilities = ModelCapabilities()

    def __init__(self, *, device="auto", cuda_graph=None):
        self.device = resolve_device(device)
        self.config = cuda_graph if isinstance(cuda_graph, CUDAGraphConfig) else CUDAGraphConfig(
            enabled="auto" if cuda_graph is None else cuda_graph
        )
        self._signature = None
        self._warned = set()
        self._status = dict(mode="not-run", calls=0, graph_calls=0, eager_calls=0,
                            invalidations=0, reason=None)

    def graph_unavailable_reason(self, properties):
        if self.device.type != "cuda":
            return "CUDA Graph requires a CUDA device"
        if not properties <= self.capabilities.cuda_graph_properties:
            return f"CUDA Graph does not support requested properties: {sorted(properties)}"
        return None

    def predict(self, atoms, properties):
        properties = frozenset(properties) - {"free_energy"}
        properties |= {"energy", "forces"}
        unsupported = properties - self.capabilities.properties
        if unsupported:
            raise PropertyNotImplementedError(f"Unsupported properties: {sorted(unsupported)}")
        if len(atoms) == 0:
            raise ValueError("Cannot evaluate an empty Atoms object")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError("Positions and cell must be finite")
        if self.capabilities.periodic_only and (not atoms.pbc.all() or atoms.cell.volume <= 0):
            raise ValueError("This backend requires a nonzero, fully periodic cell (pbc=True)")
        signature = (atoms.numbers.tobytes(), atoms.cell.array.tobytes(), atoms.pbc.tobytes())
        if signature != self._signature:
            self.clear_graphs()
            if self._signature is not None:
                self._status["invalidations"] += 1
            self._signature = signature
        reason = "disabled by configuration" if self.config.enabled is False else self.graph_unavailable_reason(properties)
        if reason and self.config.enabled is True:
            raise RuntimeError(f"CUDA Graph was required but is unavailable: {reason}")
        if reason and self.config.enabled == "auto" and reason not in self._warned:
            warnings.warn(f"fastMD uses eager inference: {reason}", RuntimeWarning, stacklevel=2)
            self._warned.add(reason)
        mode = "eager" if reason else "cuda_graph"
        context = torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()
        # Inference still needs autograd for energy derivatives (forces/stress).
        with context, torch.inference_mode(False), torch.enable_grad():
            results = (self._predict_eager if reason else self._predict_graph)(atoms, properties)
        self._status.update(mode=mode, reason=reason)
        self._status["calls"] += 1
        self._status["eager_calls" if reason else "graph_calls"] += 1
        return results

    @abstractmethod
    def _predict_eager(self, atoms, properties):
        """Return energy [eV], forces [eV/Å], optional ASE stress [eV/Å³]."""

    def _predict_graph(self, atoms, properties):
        raise NotImplementedError("This adapter has no CUDA Graph implementation")

    def clear_graphs(self):
        """Release model-specific captures, retaining loaded model weights."""

    def stats(self):
        return {**self._status, "device": str(self.device)}
