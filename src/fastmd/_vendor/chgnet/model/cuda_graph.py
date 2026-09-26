"""Bucketed CUDA Graph execution for fixed-cell CHGNet MD."""

from __future__ import annotations

import os
import time

import torch

from fastmd._vendor.chgnet.graph.crystalgraph import CrystalGraph

_FIELDS = (
    "atomic_number",
    "atom_frac_coord",
    "atom_graph",
    "neighbor_image",
    "directed2undirected",
    "undirected2directed",
    "bond_graph",
    "lattice",
)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def pad_crystal_graph(
    graph: CrystalGraph,
    u_max: int,
    t_max: int,
    n_dummy: int = 64,
) -> tuple[CrystalGraph, int]:
    """Pad one ragged graph to fixed edge and triplet capacities.

    Dummy atoms, edges, and triplets form a disconnected sink subgraph. The
    model readout and returned forces are restricted to the first ``n_real``
    atoms, so padding cannot contribute to the physical system.
    """
    device = graph.lattice.device
    n_real = int(graph.atomic_number.shape[0])
    n_edge = int(graph.atom_graph.shape[0])
    n_undirected = int(graph.undirected2directed.shape[0])
    n_triplet = int(graph.bond_graph.shape[0])
    e_max = 2 * int(u_max)
    n_pad_u = int(u_max) - n_undirected

    if n_dummy <= 0:
        raise ValueError("n_dummy must be positive")
    if n_edge != 2 * n_undirected:
        raise ValueError("CUDA Graph padding requires two directed edges per bond")
    if n_edge > e_max or n_pad_u < 1 or n_triplet > t_max:
        raise ValueError(
            "graph does not fit CUDA Graph capacity: "
            f"E={n_edge}/{e_max}, U={n_undirected}/{u_max}, "
            f"T={n_triplet}/{t_max}"
        )

    atomic_number = torch.cat(
        (graph.atomic_number, graph.atomic_number.new_ones(n_dummy))
    )
    atom_frac_coord = torch.cat(
        (graph.atom_frac_coord, graph.atom_frac_coord.new_zeros(n_dummy, 3))
    )

    e_pad = e_max - n_edge
    pad_edge_pair = torch.arange(e_pad, device=device) // 2
    pad_edge_sink = n_real + (pad_edge_pair * n_dummy // n_pad_u)
    atom_graph = graph.atom_graph.new_zeros((e_max, 2))
    atom_graph[:n_edge] = graph.atom_graph
    atom_graph[n_edge:, 0] = pad_edge_sink
    atom_graph[n_edge:, 1] = pad_edge_sink

    neighbor_image = graph.neighbor_image.new_zeros((e_max, 3))
    neighbor_image[:n_edge] = graph.neighbor_image
    # A non-zero image gives every dummy self-edge a finite, non-zero vector.
    neighbor_image[n_edge:, 0] = 3.0

    pad_u_ids = torch.arange(n_undirected, u_max, device=device)
    directed2undirected = graph.directed2undirected.new_zeros(e_max)
    directed2undirected[:n_edge] = graph.directed2undirected
    directed2undirected[n_edge:] = pad_u_ids.repeat_interleave(2)
    undirected2directed = graph.undirected2directed.new_zeros(u_max)
    undirected2directed[:n_undirected] = graph.undirected2directed
    undirected2directed[n_undirected:] = torch.arange(n_edge, e_max, 2, device=device)

    bond_graph = graph.bond_graph.new_zeros((t_max, 5))
    if n_triplet:
        bond_graph[:n_triplet] = graph.bond_graph
    t_pad = t_max - n_triplet
    pad_u = torch.arange(t_pad, device=device) % n_pad_u
    bond_graph[n_triplet:, 0] = n_real + (torch.arange(t_pad, device=device) % n_dummy)
    bond_graph[n_triplet:, 1] = n_undirected + pad_u
    bond_graph[n_triplet:, 2] = n_edge + 2 * pad_u
    bond_graph[n_triplet:, 3] = n_undirected + pad_u
    bond_graph[n_triplet:, 4] = n_edge + 2 * pad_u

    padded = CrystalGraph(
        atomic_number=atomic_number,
        atom_frac_coord=atom_frac_coord,
        atom_graph=atom_graph,
        neighbor_image=neighbor_image,
        directed2undirected=directed2undirected,
        undirected2directed=undirected2directed,
        bond_graph=bond_graph,
        lattice=graph.lattice,
        graph_id=None,
        mp_id=None,
        composition=graph.composition,
        atom_graph_cutoff=graph.atom_graph_cutoff,
        bond_graph_cutoff=graph.bond_graph_cutoff,
    )
    return padded, n_real


def _cap_for(
    n_undirected: int,
    n_triplet: int,
    u_step: int,
    t_step: int,
    min_pad_u: int,
) -> tuple[int, int]:
    """Return a capacity bucket with at least one padding bond."""
    if u_step <= 0 or t_step <= 0 or min_pad_u < 1:
        raise ValueError("CUDA Graph bucket sizes must be positive")
    u_cap = ((n_undirected // u_step) + 1) * u_step
    if u_cap - n_undirected < min_pad_u:
        u_cap += u_step
    t_cap = ((n_triplet // t_step) + 1) * t_step
    return int(u_cap), int(t_cap)


def _smallest_fitting_key(
    keys,
    n_undirected: int,
    n_triplet: int,
    u_step: int,
    t_step: int,
):
    """Return the cached bucket with the least normalized padding."""
    fitting = [key for key in keys if key[0] > n_undirected and key[1] >= n_triplet]
    if not fitting:
        return None
    return min(
        fitting,
        key=lambda key: (
            (key[0] - n_undirected) / u_step + (key[1] - n_triplet) / t_step,
            key[0],
            key[1],
        ),
    )


class BucketedGraphRunner:
    """Capture and replay model graphs for one fixed-composition system."""

    def __init__(
        self,
        model,
        *,
        task: str = "ef",
        u_step: int = 128,
        t_step: int = 1024,
        n_dummy: int = 64,
        min_pad_u: int = 64,
        warmup: int = 3,
    ) -> None:
        if task != "ef":
            raise ValueError("BucketedGraphRunner currently supports task='ef' only")
        if not getattr(model, "mlp_first", False):
            raise ValueError("CUDA Graph padding currently requires mlp_first=True")
        if not torch.cuda.is_available():
            raise ValueError("BucketedGraphRunner requires CUDA")

        self.model = model
        self.task = task
        self.u_step = u_step
        self.t_step = t_step
        self.n_dummy = n_dummy
        self.min_pad_u = min_pad_u
        self.warmup = warmup
        if self.warmup < 1:
            raise ValueError("CUDA Graph capture warmup must be positive")

        self.pool = torch.cuda.graph_pool_handle()
        self.cache: dict[tuple[int, int], dict] = {}
        self.captures = 0
        self.capture_events: list[dict[str, int | float]] = []
        self.phase = "setup"
        self.run_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_fit_reuses = 0
        self.n_real: int | None = None

    def _composition_energy(self, graph: CrystalGraph) -> torch.Tensor | int:
        composition_model = self.model.composition_model
        if composition_model is None:
            return 0
        with torch.no_grad():
            energy = composition_model([graph])
        return energy.detach()

    def _capture(
        self,
        u_cap: int,
        t_cap: int,
        sample_graph: CrystalGraph,
    ) -> None:
        u_real = int(sample_graph.undirected2directed.shape[0])
        t_real = int(sample_graph.bond_graph.shape[0])
        torch.cuda.synchronize()
        started = time.perf_counter()
        padded, n_real = pad_crystal_graph(sample_graph, u_cap, t_cap, self.n_dummy)
        composition_energy = self._composition_energy(sample_graph)

        current_stream = torch.cuda.current_stream(sample_graph.lattice.device)
        side_stream = torch.cuda.Stream(device=sample_graph.lattice.device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            for _ in range(self.warmup):
                self.model(
                    [padded],
                    task=self.task,
                    n_real=n_real,
                    composition_energy=composition_energy,
                )
        current_stream.wait_stream(side_stream)

        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph, pool=self.pool):
            output = self.model(
                [padded],
                task=self.task,
                n_real=n_real,
                composition_energy=composition_energy,
            )
        torch.cuda.synchronize()

        event: dict[str, int | float] = {
            "capture_id": self.captures + 1,
            "u_real": u_real,
            "t_real": t_real,
            "u_cap": int(u_cap),
            "t_cap": int(t_cap),
            "n_real": int(n_real),
            "n_total": int(n_real + self.n_dummy),
            "setup_s": time.perf_counter() - started,
        }
        self.cache[(u_cap, t_cap)] = {
            "padded": padded,
            "graph": cuda_graph,
            "output": output,
            "n_real": n_real,
            # CUDA Graph kernels retain this tensor's device pointer.
            "composition_energy": composition_energy,
            "capture": event,
        }
        self.capture_events.append(event)
        self.captures += 1
        print(
            "[cuda-graph][capture] "
            f"id={event['capture_id']} phase={self.phase} "
            f"real=(U={u_real},T={t_real}) "
            f"bucket=(U={u_cap},T={t_cap}) "
            f"setup={event['setup_s']:.3f}s",
            flush=True,
        )

    def _bucket(self, graph: CrystalGraph) -> tuple[int, int]:
        return _cap_for(
            int(graph.undirected2directed.shape[0]),
            int(graph.bond_graph.shape[0]),
            self.u_step,
            self.t_step,
            self.min_pad_u,
        )

    def ensure(self, sample_graph: CrystalGraph, key: tuple[int, int]) -> None:
        """Capture a capacity bucket if it is not already cached."""
        if key not in self.cache:
            self._capture(key[0], key[1], sample_graph)

    def precapture(self, sample_graph: CrystalGraph, *, proactive: bool = True) -> None:
        """Capture the current bucket and optional neighboring capacities."""
        u_cap, t_cap = self._bucket(sample_graph)
        keys = [(u_cap, t_cap)]
        if proactive:
            keys.extend(
                (
                    (u_cap + self.u_step, t_cap),
                    (u_cap, t_cap + self.t_step),
                    (u_cap + self.u_step, t_cap + self.t_step),
                )
            )
        for key in keys:
            self.ensure(sample_graph, key)

    def reset_run_stats(self, phase: str = "production") -> None:
        """Reset hit/miss counters without discarding captured graphs."""
        self.phase = phase
        self.run_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_fit_reuses = 0

    def stats(self) -> dict[str, object]:
        """Return capture and replay statistics."""
        return {
            "captures": self.captures,
            "cached_buckets": [list(key) for key in sorted(self.cache)],
            "run_calls": self.run_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_fit_reuses": self.cache_fit_reuses,
            "capture_events": [event.copy() for event in self.capture_events],
        }

    def run(self, graph: CrystalGraph) -> tuple[dict, int]:
        """Replay the smallest fitting graph; outputs live until the next replay."""
        n_real = int(graph.atomic_number.shape[0])
        if self.n_real is None:
            self.n_real = n_real
        elif n_real != self.n_real:
            raise ValueError("BucketedGraphRunner requires a fixed atom count")

        self.run_calls += 1
        n_undirected = int(graph.undirected2directed.shape[0])
        n_triplet = int(graph.bond_graph.shape[0])
        canonical_key = _cap_for(
            n_undirected,
            n_triplet,
            self.u_step,
            self.t_step,
            self.min_pad_u,
        )
        key = _smallest_fitting_key(
            self.cache,
            n_undirected,
            n_triplet,
            self.u_step,
            self.t_step,
        )
        if key is None:
            self.cache_misses += 1
            key = canonical_key
            self.ensure(graph, key)
        else:
            self.cache_hits += 1
            if key != canonical_key:
                self.cache_fit_reuses += 1

        entry = self.cache[key]
        with torch.no_grad():
            new_padded, _ = pad_crystal_graph(graph, key[0], key[1], self.n_dummy)
            for field in _FIELDS:
                getattr(entry["padded"], field).copy_(getattr(new_padded, field))
        entry["graph"].replay()
        return entry["output"], entry["n_real"]
