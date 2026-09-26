"""Lazy model registration: importing fastmd does not import DGL/pymatgen."""
from importlib import import_module

_FACTORIES = {
    "matris": "fastmd.models.matris:MatRISModel",
    "chgnet": "fastmd.models.chgnet:CHGNetModel",
    "alignn": "fastmd.models.alignn:ALIGNNModel",
}


def available_models():
    return tuple(sorted(_FACTORIES))


def register_model(name, factory, *, overwrite=False):
    """Register a ModelBackend factory or 'module:Class' import path."""
    name = name.lower().strip()
    if not name or not (callable(factory) or isinstance(factory, str) and ":" in factory):
        raise ValueError("Supply a nonempty name and callable or 'module:Class' factory")
    if name in _FACTORIES and not overwrite:
        raise ValueError(f"Model {name!r} is already registered")
    _FACTORIES[name] = factory


def load_model(name, **kwargs):
    key = name.lower()
    if key not in _FACTORIES:
        raise ValueError(f"Unknown model {name!r}; available models: {', '.join(available_models())}")
    factory = _FACTORIES[key]
    try:
        if isinstance(factory, str):
            module, attribute = factory.split(":", 1)
            factory = getattr(import_module(module), attribute)
        return factory(**kwargs)
    except ImportError as exc:
        raise ImportError(
            f"Cannot load {name}: {exc}. Install fastMD's [{key}] extra and check its runtime dependencies; see README.md."
        ) from exc
