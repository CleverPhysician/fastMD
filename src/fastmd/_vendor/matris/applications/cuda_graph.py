"""Bucketed CUDA-graph execution for MatRIS MD (static-shape forward+backward).

The model forward is D2H-sync-free, so a fixed-shape forward+backward can be
captured with torch.cuda.graph and replayed each MD step (collapsing ~3300
kernel launches into one replay -> removes launch-gap GPU idle).

MD edge/triplet counts fluctuate, so shapes are made static by padding to fixed
CAPACITY BUCKETS: padding edges/triplets are routed to dummy SINK atoms (index
>= N) that are masked out of the energy/reference via the model's n_real arg, so
results are numerically identical to the ragged model on the real atoms. One
graph is captured per (U_cap, T_cap) bucket and selected at replay time from the
current real counts (host-known .shape, no sync); MD counts cluster into a few
buckets, so after warmup every step is a cache hit.

Note: this does NOT make the ragged neighbor-graph BUILD sync-free (the sparse
three-body line graph cannot be densified without exploding compute); it removes
the forward's launch-gap idle, which is the dominant, addressable component.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import time

import torch

from .._nvtx import nvtx_range
from ..graph.radiusgraph import RadiusGraph

_FIELDS = ["atomic_number", "atom_frac_coord", "atom_graph", "neighbor_image",
           "directed2undirected", "undirected2directed", "line_graph", "lattice"]


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _graph_input_bytes(graph: RadiusGraph) -> int:
    return sum(_tensor_bytes(getattr(graph, field)) for field in _FIELDS)


def _memory_snapshot() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "device_used_bytes": total - free,
        "device_total_bytes": total,
    }


def pad_radius_graph(g: RadiusGraph, u_max: int, t_max: int, n_dummy: int = 64):
    """Pad a single ragged RadiusGraph to fixed capacity (e_max=2*u_max, t_max)
    with n_dummy dummy SINK atoms. Returns (padded_graph, n_real).

    Padding self-loops use a large periodic-image shift so lengths are finite
    (radial basis -> 0) and unit vectors well-defined. Padding edges/triplets are
    spread across the dummy atoms and padding undirected edges to avoid atomic-add
    scatter contention.
    """
    dev = g.lattice.device
    n = g.atomic_number.shape[0]
    e = g.atom_graph.shape[0]
    u = g.undirected2directed.shape[0]
    t = g.line_graph.shape[0]
    e_max = 2 * u_max
    n_pad_u = u_max - u
    assert e <= e_max and n_pad_u >= 1 and t <= t_max, (e, u, t, e_max, u_max, t_max)
    D = n_dummy  # fixed -> total atom count (N+D) is static across steps

    atomic_number = torch.cat([g.atomic_number, g.atomic_number.new_ones(D)])
    atom_frac_coord = torch.cat([g.atom_frac_coord, g.atom_frac_coord.new_zeros(D, 3)])

    e_pad = e_max - e
    pad_edge_pair = torch.arange(e_pad, device=dev) // 2
    pad_edge_sink = n + (pad_edge_pair * D // n_pad_u)
    atom_graph = g.atom_graph.new_zeros(e_max, 2)
    atom_graph[:e] = g.atom_graph
    atom_graph[e:, 0] = pad_edge_sink
    atom_graph[e:, 1] = pad_edge_sink

    neighbor_image = g.neighbor_image.new_zeros(e_max, 3)
    neighbor_image[:e] = g.neighbor_image
    neighbor_image[e:] = 3.0

    pad_u_ids = torch.arange(u, u_max, device=dev)
    directed2undirected = g.directed2undirected.new_zeros(e_max)
    directed2undirected[:e] = g.directed2undirected
    directed2undirected[e:] = pad_u_ids.repeat_interleave(2)
    undirected2directed = g.undirected2directed.new_zeros(u_max)
    undirected2directed[:u] = g.undirected2directed
    undirected2directed[u:] = torch.arange(e, e_max, 2, device=dev)

    line_graph = g.line_graph.new_zeros(t_max, 5)
    if t > 0:
        line_graph[:t] = g.line_graph
    t_pad = t_max - t
    pu = torch.arange(t_pad, device=dev) % n_pad_u
    line_graph[t:, 0] = n + (torch.arange(t_pad, device=dev) % D)
    line_graph[t:, 1] = u + pu
    line_graph[t:, 2] = e + 2 * pu
    line_graph[t:, 3] = u + pu
    line_graph[t:, 4] = e + 2 * pu

    pg = RadiusGraph(
        atomic_number=atomic_number, atom_frac_coord=atom_frac_coord,
        atom_graph=atom_graph, neighbor_image=neighbor_image,
        directed2undirected=directed2undirected, undirected2directed=undirected2directed,
        line_graph=line_graph, lattice=g.lattice, graph_id=None, mp_id=None,
        composition=g.composition, atom_graph_cutoff=g.atom_graph_cutoff,
        line_graph_cutoff=g.line_graph_cutoff,
        atom_target_sorted=getattr(g, "atom_target_sorted", False),
    )
    return pg, n


@dataclass
class _PaddedGraphWorkspace:
    """Reusable padded inputs for one captured CUDA Graph capacity bucket."""

    pg: RadiusGraph
    n_real: int
    u_max: int
    t_max: int
    n_dummy: int
    edge_pair: torch.Tensor
    u_ids: torch.Tensor
    twice_u_ids: torch.Tensor
    triplet_ids: torch.Tensor

    @classmethod
    def create(
        cls,
        pg: RadiusGraph,
        n_real: int,
        u_max: int,
        t_max: int,
        n_dummy: int,
    ) -> "_PaddedGraphWorkspace":
        device = pg.lattice.device
        edge_pair = (
            torch.arange(2 * u_max, device=device, dtype=pg.atom_graph.dtype) // 2
        )
        u_ids = torch.arange(
            u_max, device=device, dtype=pg.undirected2directed.dtype
        )
        return cls(
            pg=pg,
            n_real=n_real,
            u_max=u_max,
            t_max=t_max,
            n_dummy=n_dummy,
            edge_pair=edge_pair,
            u_ids=u_ids,
            twice_u_ids=2 * u_ids,
            triplet_ids=torch.arange(
                t_max, device=device, dtype=pg.line_graph.dtype
            ),
        )

    def update(self, graph: RadiusGraph) -> None:
        """Update real rows and regenerate padding without new allocations."""
        with nvtx_range("static input copy"):
            pg = self.pg
            n = graph.atomic_number.shape[0]
            e = graph.atom_graph.shape[0]
            u = graph.undirected2directed.shape[0]
            t = graph.line_graph.shape[0]
            e_max = 2 * self.u_max
            n_pad_u = self.u_max - u
            assert n == self.n_real, (n, self.n_real)
            assert e <= e_max and n_pad_u >= 1 and t <= self.t_max, (
                e,
                u,
                t,
                e_max,
                self.u_max,
                self.t_max,
            )

            pg.atomic_number[:n].copy_(graph.atomic_number)
            pg.atom_frac_coord[:n].copy_(graph.atom_frac_coord)
            pg.lattice.copy_(graph.lattice)
            pg.atom_graph[:e].copy_(graph.atom_graph)
            pg.neighbor_image[:e].copy_(graph.neighbor_image)
            pg.directed2undirected[:e].copy_(graph.directed2undirected)
            pg.undirected2directed[:u].copy_(graph.undirected2directed)
            if t > 0:
                pg.line_graph[:t].copy_(graph.line_graph)

            if e < e_max:
                e_pad = e_max - e
                pad_edge_sink = n + (
                    self.edge_pair[:e_pad] * self.n_dummy // n_pad_u
                )
                pg.atom_graph[e:, 0].copy_(pad_edge_sink)
                pg.atom_graph[e:, 1].copy_(pad_edge_sink)
                pg.neighbor_image[e:].fill_(3.0)
                pg.directed2undirected[e:e_max:2].copy_(
                    self.u_ids[u:self.u_max]
                )
                pg.directed2undirected[e + 1:e_max:2].copy_(
                    self.u_ids[u:self.u_max]
                )
                pg.undirected2directed[u:self.u_max].copy_(
                    self.twice_u_ids[u:self.u_max]
                )

            if t < self.t_max:
                triplet_ids = self.triplet_ids[: self.t_max - t]
                padding_pair = triplet_ids % n_pad_u
                pg.line_graph[t:, 0].copy_(
                    n + (triplet_ids % self.n_dummy)
                )
                pg.line_graph[t:, 1].copy_(u + padding_pair)
                pg.line_graph[t:, 2].copy_(e + 2 * padding_pair)
                pg.line_graph[t:, 3].copy_(u + padding_pair)
                pg.line_graph[t:, 4].copy_(e + 2 * padding_pair)


def _cap_for(u, t, u_step, t_step, min_pad_u):
    u_cap = ((u // u_step) + 1) * u_step
    if u_cap - u < min_pad_u:
        u_cap += u_step
    t_cap = ((t // t_step) + 1) * t_step
    return int(u_cap), int(t_cap)


def _smallest_fitting_key(keys, u, t, u_step, t_step):
    """Return the cached bucket with the least normalized padding."""
    fitting = [key for key in keys if key[0] > u and key[1] >= t]
    if not fitting:
        return None
    return min(
        fitting,
        key=lambda key: (
            (key[0] - u) / u_step + (key[1] - t) / t_step,
            key[0],
            key[1],
        ),
    )


class BucketedGraphRunner:
    """Captures one CUDA graph per capacity bucket; replays the matching one."""

    def __init__(self, model, task="efsm", u_step=512, t_step=8192,
                 n_dummy=64, min_pad_u=256, warmup=3,
                 enable_model_fusions=True):
        self.model = model
        self.task = task
        self.u_step = u_step
        self.t_step = t_step
        self.n_dummy = n_dummy
        self.min_pad_u = min_pad_u
        self.warmup = warmup
        self.profile_stages = _env_bool(
            "MATRIS_CUDA_GRAPH_PROFILE_STAGES", False
        )
        self.static_workspace = _env_bool(
            "MATRIS_CUDA_GRAPH_STATIC_WORKSPACE", True
        )
        self._profile_events = None
        # Mutually exclusive bucket replays can safely share one graph pool.
        self.pool = torch.cuda.graph_pool_handle()
        # Generic Triton fusions are enabled only inside warm-up/capture so the
        # dynamic path can retain its independently tunable eager thresholds.
        self.enable_model_fusions = bool(enable_model_fusions)
        self._capture_exact_capacity = False
        self.cache_fit_reuses = 0
        self.cache = {}
        self.captures = 0
        self.capture_events = []
        self.phase = "setup"
        self.run_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_run_info = None

    def _capture(self, u_cap, t_cap, sample_g):
        u_real = int(sample_g.undirected2directed.shape[0])
        t_real = int(sample_g.line_graph.shape[0])
        e_real = int(sample_g.atom_graph.shape[0])
        torch.cuda.synchronize()
        memory_before = _memory_snapshot()
        setup_start = time.perf_counter()
        pg, n_real = pad_radius_graph(sample_g, u_cap, t_cap, self.n_dummy)
        workspace = _PaddedGraphWorkspace.create(
            pg=pg,
            n_real=n_real,
            u_max=u_cap,
            t_max=t_cap,
            n_dummy=self.n_dummy,
        )
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        from ..model.interaction_block import (
            restore_fused_graph_optimizations,
            set_fused_graph_optimizations,
        )
        from ..model.feature_embed import (
            restore_fused_feature_graph_optimizations,
            set_fused_feature_graph_optimizations,
        )

        previous = set_fused_graph_optimizations(self.enable_model_fusions)
        previous_feature = set_fused_feature_graph_optimizations(
            self.enable_model_fusions
        )
        try:
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self.model([pg], task=self.task, is_training=False, n_real=n_real)
            torch.cuda.current_stream().wait_stream(side)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, pool=self.pool):
                out = self.model([pg], task=self.task, is_training=False, n_real=n_real)
        finally:
            restore_fused_feature_graph_optimizations(previous_feature)
            restore_fused_graph_optimizations(previous)
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - setup_start
        memory_after = _memory_snapshot()
        event = {
            "capture_id": self.captures + 1,
            "phase": self.phase,
            "run_call": self.run_calls,
            "u_real": u_real,
            "t_real": t_real,
            "e_real": e_real,
            "u_cap": int(u_cap),
            "t_cap": int(t_cap),
            "e_cap": int(2 * u_cap),
            "n_real": int(n_real),
            "n_total": int(n_real + self.n_dummy),
            "pad_u": int(u_cap - u_real),
            "pad_t": int(t_cap - t_real),
            "static_input_bytes": _graph_input_bytes(pg),
            "output_bytes": _tensor_bytes(out),
            "setup_s": setup_s,
            "allocated_delta_bytes": (
                memory_after["allocated_bytes"] - memory_before["allocated_bytes"]
            ),
            "reserved_delta_bytes": (
                memory_after["reserved_bytes"] - memory_before["reserved_bytes"]
            ),
            "device_used_delta_bytes": (
                memory_after["device_used_bytes"] - memory_before["device_used_bytes"]
            ),
            **{f"after_{k}": v for k, v in memory_after.items()},
        }
        self.cache[(u_cap, t_cap)] = dict(
            pg=pg,
            graph=gr,
            out=out,
            n_real=n_real,
            workspace=workspace,
            capture=event,
        )
        self.capture_events.append(event)
        self.captures += 1
        mib = 1024**2
        print(
            "[cuda-graph][capture] "
            f"id={event['capture_id']} phase={self.phase} "
            f"real=(U={u_real},T={t_real}) "
            f"bucket=(U={u_cap},T={t_cap},E={2*u_cap}) "
            f"pad=(U={event['pad_u']},T={event['pad_t']}) "
            f"static_inputs={event['static_input_bytes']/mib:.2f}MiB "
            f"setup={setup_s:.3f}s "
            f"alloc_delta={event['allocated_delta_bytes']/mib:+.2f}MiB "
            f"reserved_delta={event['reserved_delta_bytes']/mib:+.2f}MiB "
            f"device_used_delta={event['device_used_delta_bytes']/mib:+.2f}MiB",
            flush=True,
        )

    def _bucket(self, g: RadiusGraph):
        u, t = g.undirected2directed.shape[0], g.line_graph.shape[0]
        return _cap_for(u, t, self.u_step, self.t_step, self.min_pad_u)

    def _cached_bucket_that_fits(self, u: int, t: int):
        """Return the least-work cached bucket that safely covers this graph.

        Capacity buckets overlap. Requiring an exact canonical key needlessly
        captures a second graph when counts move downward across a bucket
        boundary even though a larger cached graph already fits.
        """
        return _smallest_fitting_key(
            self.cache, u, t, self.u_step, self.t_step
        )

    @contextmanager
    def exact_capacity_capture(self):
        """Temporarily capture canonical buckets instead of covering buckets."""
        previous = self._capture_exact_capacity
        self._capture_exact_capacity = True
        try:
            yield
        finally:
            self._capture_exact_capacity = previous

    def ensure(self, sample_g: RadiusGraph, key):
        """Capture the (u_cap, t_cap) bucket if not already cached. sample_g must
        fit (its real counts <= capacity)."""
        if key not in self.cache:
            self._capture(key[0], key[1], sample_g)

    def precapture(self, sample_g: RadiusGraph, proactive: bool = True):
        """Pre-capture the bucket for sample_g and (optionally) the next-larger U/T
        neighbor buckets, without replaying — so later run() calls are pure replay.
        Larger buckets always fit sample_g (just more padding)."""
        u0, t0 = self._bucket(sample_g)
        keys = [(u0, t0)]
        if proactive:
            keys += [(u0 + self.u_step, t0),
                     (u0, t0 + self.t_step),
                     (u0 + self.u_step, t0 + self.t_step)]
        for k in keys:
            self.ensure(sample_g, k)

    def reset_run_stats(self, phase: str = "production") -> None:
        """Reset hit/miss counters without discarding captured graphs."""
        self.phase = phase
        self.run_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_run_info = None

    def cache_report(self) -> list[dict]:
        """Return JSON/TSV-friendly metadata for every cached graph."""
        return [self.cache[key]["capture"].copy() for key in sorted(self.cache)]

    def stats(self) -> dict[str, object]:
        return {
            "captures": self.captures,
            "cache_fit_reuses": self.cache_fit_reuses,
            "cached_buckets": [list(key) for key in sorted(self.cache)],
            "enable_model_fusions": self.enable_model_fusions,
            "static_workspace": self.static_workspace,
            "shared_graph_pool": True,
            "run_calls": self.run_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def print_cache_summary(self, label: str = "cache") -> None:
        """Print bucket capacities and current CUDA memory outside timed regions."""
        torch.cuda.synchronize()
        memory = _memory_snapshot()
        gib = 1024**3
        keys = ", ".join(f"({u},{t})" for u, t in sorted(self.cache)) or "none"
        hit_rate = self.cache_hits / self.run_calls if self.run_calls else 0.0
        print(
            f"[cuda-graph][{label}] graphs={len(self.cache)} keys=[{keys}] "
            f"runs={self.run_calls} hits={self.cache_hits} misses={self.cache_misses} "
            f"hit_rate={hit_rate:.4f} "
            f"allocated={memory['allocated_bytes']/gib:.3f}GiB "
            f"reserved={memory['reserved_bytes']/gib:.3f}GiB "
            f"device_used={memory['device_used_bytes']/gib:.3f}GiB",
            flush=True,
        )

    def synchronize_profile(self) -> None:
        """Finalize optional CUDA-event stage timings after replay."""
        if self._profile_events is None or self.last_run_info is None:
            return
        events = self._profile_events
        events["replay_done"].synchronize()
        self.last_run_info.update(
            staticize_copy_cuda_ms=events["static_begin"].elapsed_time(
                events["static_done"]
            ),
            replay_cuda_ms=events["static_done"].elapsed_time(
                events["replay_done"]
            ),
        )
        self._profile_events = None

    def run(self, g: RadiusGraph):
        """Return (out_dict, n_real). out_dict tensors are valid until the next run().

        NB: grad must stay enabled — the model computes conservative forces via an
        internal torch.autograd.grad; the copy_ of inputs is wrapped in no_grad.
        """
        self.run_calls += 1
        u, t = int(g.undirected2directed.shape[0]), int(g.line_graph.shape[0])
        e = int(g.atom_graph.shape[0])
        canonical_key = _cap_for(
            u, t, self.u_step, self.t_step, self.min_pad_u
        )
        if self._capture_exact_capacity:
            key = canonical_key if canonical_key in self.cache else None
        else:
            key = self._cached_bucket_that_fits(u, t)
        cache_hit = key is not None
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            key = canonical_key
            if key not in self.cache:
                self._capture(key[0], key[1], g)
        assert key is not None
        if cache_hit and key != canonical_key:
            self.cache_fit_reuses += 1
        ent = self.cache[key]
        if self.profile_stages:
            static_begin = torch.cuda.Event(enable_timing=True)
            static_done = torch.cuda.Event(enable_timing=True)
            replay_done = torch.cuda.Event(enable_timing=True)
            static_begin.record()
        with torch.no_grad():
            if self.static_workspace:
                ent["workspace"].update(g)
            else:
                newpg, _ = pad_radius_graph(g, key[0], key[1], self.n_dummy)
                for field in _FIELDS:
                    getattr(ent["pg"], field).copy_(getattr(newpg, field))
        if self.profile_stages:
            static_done.record()
        replay_start = time.perf_counter()
        ent["graph"].replay()
        replay_enqueue_ms = (time.perf_counter() - replay_start) * 1000
        if self.profile_stages:
            replay_done.record()
            self._profile_events = {
                "static_begin": static_begin,
                "static_done": static_done,
                "replay_done": replay_done,
            }
        self.last_run_info = {
            "phase": self.phase,
            "run_call": self.run_calls,
            "cache_hit": cache_hit,
            "capture_id": ent["capture"]["capture_id"],
            "cache_size": len(self.cache),
            "captures": self.captures,
            "u_real": u,
            "t_real": t,
            "e_real": e,
            "u_cap": key[0],
            "t_cap": key[1],
            "e_cap": 2 * key[0],
            "canonical_u_cap": canonical_key[0],
            "canonical_t_cap": canonical_key[1],
            "covering_reuse": cache_hit and key != canonical_key,
            "pad_u": key[0] - u,
            "pad_t": key[1] - t,
            "n_real": ent["n_real"],
            "n_total": ent["n_real"] + self.n_dummy,
            "static_input_bytes": ent["capture"]["static_input_bytes"],
            "replay_enqueue_ms": replay_enqueue_ms,
        }
        return ent["out"], ent["n_real"]
