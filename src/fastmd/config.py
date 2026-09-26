"""Instance-local settings; ordinary ASE users need none of these knobs."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CUDAGraphConfig:
    """auto: use supported captures; True: require capture; False: eager only.

    Capacity defaults are model-specific. Edges in MatRIS/CHGNet are undirected;
    ALIGNN uses directed edges. The cache is cleared when its bucket limit is
    exceeded. Warmup never moves the user's atoms or starts a trial trajectory.
    """

    enabled: bool | Literal["auto"] = "auto"
    warmup_steps: int = 3
    max_cached_graphs: int = 8
    edge_capacity_step: int | None = None
    triplet_capacity_step: int | None = None
    enable_fusions: bool = True

    def __post_init__(self):
        if not (type(self.enabled) is bool or self.enabled == "auto"):
            raise ValueError("enabled must be True, False, or 'auto'")
        for name in ("warmup_steps", "max_cached_graphs", "edge_capacity_step", "triplet_capacity_step"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
