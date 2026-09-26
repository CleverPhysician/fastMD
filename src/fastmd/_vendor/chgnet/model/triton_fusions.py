"""Context-local switches for graph-safe CHGNet Triton fusions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_MODEL_FUSIONS_ENABLED: ContextVar[bool] = ContextVar(
    "chgnet_model_fusions_enabled",
    default=False,
)


def model_fusions_enabled() -> bool:
    """Return whether graph-safe model fusions are enabled in this context."""
    return _MODEL_FUSIONS_ENABLED.get()


def set_model_fusions_enabled(enabled: bool) -> bool:
    """Set the context-local fusion switch and return its previous value."""
    previous = _MODEL_FUSIONS_ENABLED.get()
    _MODEL_FUSIONS_ENABLED.set(bool(enabled))
    return previous


def prepare_model_fusions(model) -> int:
    """Prepack every supported frozen GatedMLP before CUDA Graph capture."""
    prepared = 0
    for module in model.modules():
        prepare = getattr(module, "prepare_fused_inference", None)
        if prepare is not None:
            prepared += int(prepare())
    return prepared


@contextmanager
def model_fusion_mode(enabled: bool):
    """Temporarily enable or disable graph-safe model fusions."""
    previous = set_model_fusions_enabled(enabled)
    try:
        yield
    finally:
        set_model_fusions_enabled(previous)
