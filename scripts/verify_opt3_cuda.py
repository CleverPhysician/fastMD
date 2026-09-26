#!/usr/bin/env python3
"""Small CUDA smoke for the fastMD opt3 fixed-cell relaxation path.

This is deliberately a correctness/route check, not a formal benchmark.  It
keeps model loading, graph warmup/capture, eager FIRE, and whole-step opt3
timings separate so the result cannot be mistaken for the earlier A100 tables.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ase.build import bulk

from fastmd import FastMDCalculator, GPUFireConfig, run_gpu_fire, run_gpu_fire_graph


MODELS = ("dpa4", "nequip", "orbv3", "sevennet", "tace")


def _parse_checkpoints(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in MODELS or not path:
            raise ValueError(f"checkpoint must look like model=/path, got {value!r}")
        result[name] = str(Path(path).expanduser())
    return result


def _fresh_atoms():
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions = atoms.positions + 0.01
    return atoms


def _sync(device):
    torch.cuda.synchronize(device)


def _initial_device_output(callback, positions):
    _sync(positions.device)
    started = time.perf_counter()
    energy, forces = callback(positions)
    _sync(positions.device)
    return energy.detach().clone(), forces.detach().clone(), time.perf_counter() - started


def verify_one(name, checkpoint, device, steps, warmup_steps):
    atoms = _fresh_atoms()
    row = {"model": name, "checkpoint": checkpoint, "status": "failed"}
    eager = None
    opt3 = None
    try:
        eager = FastMDCalculator(name, checkpoint=checkpoint, device=device, cuda_graph=False)
        graph_enabled = name != "tace"
        opt3 = FastMDCalculator(
            name,
            checkpoint=checkpoint,
            device=device,
            cuda_graph=graph_enabled,
        )
        eager_callback = eager.backend.device_callback(atoms)
        opt3_callback = opt3.backend.device_callback(atoms)
        positions = torch.as_tensor(atoms.positions, dtype=torch.float64, device=device)

        eager_energy, eager_forces, eager_call_seconds = _initial_device_output(eager_callback, positions)
        opt3_energy, opt3_forces, opt3_call_seconds = _initial_device_output(opt3_callback, positions)
        initial_force_delta = float((eager_forces - opt3_forces).abs().max().item())
        initial_energy_delta = float((eager_energy - opt3_energy).abs().item())

        config = GPUFireConfig(fmax=0.02, steps=steps, check_interval=1)
        _sync(positions.device)
        started = time.perf_counter()
        eager_result = run_gpu_fire(positions, eager_callback, config)
        _sync(positions.device)
        eager_seconds = time.perf_counter() - started

        graph_config = GPUFireConfig(fmax=0.02, steps=steps, check_interval=max(1, min(10, steps)))
        _sync(positions.device)
        started = time.perf_counter()
        opt3_result, graph_stats = run_gpu_fire_graph(
            positions, opt3_callback, graph_config, warmup_steps=warmup_steps
        )
        _sync(positions.device)
        opt3_seconds = time.perf_counter() - started

        position_delta = float((eager_result.positions - opt3_result.positions).abs().max().item())
        force_delta = float((eager_result.forces - opt3_result.forces).abs().max().item())
        energy_delta = float(abs(float(eager_result.energy) - float(opt3_result.energy)))
        correctness_pass = (
            initial_energy_delta <= 2e-5
            and initial_force_delta <= 2e-4
            and energy_delta <= 2e-5
            and force_delta <= 2e-4
            and position_delta <= 2e-3
        )
        row.update(
            status="passed" if correctness_pass else "failed",
            correctness_pass=correctness_pass,
            device=str(device),
            gpu=torch.cuda.get_device_name(device),
            eager_route=eager.backend.stats().get("route"),
            opt3_route=opt3.backend.stats().get("route"),
            initial_energy_delta=initial_energy_delta,
            initial_force_delta=initial_force_delta,
            eager_call_seconds=eager_call_seconds,
            opt3_call_seconds=opt3_call_seconds,
            eager_relax_seconds=eager_seconds,
            opt3_relax_seconds=opt3_seconds,
            relax_speedup=(eager_seconds / opt3_seconds if opt3_seconds else None),
            position_delta=position_delta,
            force_delta=force_delta,
            energy_delta=energy_delta,
            eager_converged=eager_result.converged,
            opt3_converged=opt3_result.converged,
            eager_steps=eager_result.completed_steps,
            opt3_steps=opt3_result.completed_steps,
            graph_stats=graph_stats,
        )
    except Exception as exc:  # Keep all five model rows in the report.
        row.update(error=f"{type(exc).__name__}: {exc}")
    finally:
        del eager, opt3
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
    checkpoints = _parse_checkpoints(args.checkpoint)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    report = {
        "status": "running",
        "scope": "opt3 fixed-cell CUDA smoke; not a formal speed benchmark",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for name in MODELS:
        if name in checkpoints:
            report["rows"].append(verify_one(name, checkpoints[name], device, args.steps, args.warmup_steps))
        else:
            report["rows"].append({"model": name, "status": "missing_checkpoint"})
        args.output.write_text(json.dumps(report, indent=2))
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["rows"]) else "incomplete"
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
