"""Standard ASE calculator boundary, with no model-specific user-facing API."""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress

from .models import ModelBackend, load_model


class FastMDCalculator(Calculator):
    """Attach to atoms.calc and use ASE MD, optimizers, constraints and observers.

    model: registered name or an initialized ModelBackend.
    checkpoint: file for MatRIS/CHGNet, directory for ALIGNN.
    cuda_graph: 'auto' (default), True, False, or CUDAGraphConfig.
    """

    def __init__(self, model="matris", *, checkpoint=None, device="auto",
                 cuda_graph=None, model_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(model, ModelBackend):
            if checkpoint is not None or device != "auto" or cuda_graph is not None or model_kwargs:
                raise ValueError("Configure an initialized ModelBackend directly; do not also pass model options")
            self.backend = model
        else:
            self.backend = load_model(model, checkpoint=checkpoint, device=device,
                                      cuda_graph=cuda_graph, **(model_kwargs or {}))
        self.implemented_properties = sorted(self.backend.capabilities.properties | {"free_energy"})

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # Clear first: a failed evaluation must not leave results for old atoms.
        self.results = {}
        requested = set(properties)
        unknown = requested - set(self.implemented_properties)
        if unknown:
            raise PropertyNotImplementedError(f"Unsupported properties: {sorted(unknown)}")
        raw = self.backend.predict(self.atoms.copy(), requested)
        results = {}
        for name, value in raw.items():
            if name not in self.implemented_properties:
                continue
            array = np.array(value, dtype=float, copy=True)
            if name in {"energy", "free_energy"}:
                if array.size != 1:
                    raise ValueError(f"{name} must be a single total energy")
                results[name] = float(array.reshape(-1)[0])
            else:
                if name == "stress" and array.shape == (3, 3):
                    array = full_3x3_to_voigt_6_stress(array)
                expected = {"forces": (len(self.atoms), 3), "stress": (6,), "magmoms": (len(self.atoms),)}.get(name)
                if expected is not None and array.shape != expected:
                    raise ValueError(f"Invalid {name} shape: {array.shape}, expected {expected}")
                results[name] = array
            if not np.isfinite(array).all():
                raise ValueError(f"Model returned non-finite {name}")
        required = (requested - {"free_energy"}) | {"energy", "forces"}
        if not required <= results.keys():
            raise PropertyNotImplementedError(f"Backend omitted properties: {sorted(required - results.keys())}")
        results["free_energy"] = results["energy"]
        self.results = results

    def warmup(self, atoms, properties=("energy", "forces")):
        """Capture for the current geometry without moving atoms or running MD."""
        self.calculate(atoms, properties, all_changes)
        return self.stats()

    def stats(self):
        """Actual last execution mode plus capture/cache counters."""
        return self.backend.stats()

    def clear_cache(self):
        self.backend.clear_graphs()
        self.reset()
