"""Compare identical geometries; report setup separately from replay/eager time."""
import argparse
import time
import numpy as np
import torch
from ase.build import bulk
from ase.io import read
from fastmd import FastMDCalculator


def evaluate(calc, frames):
    torch.cuda.synchronize(calc.backend.device)
    start = time.perf_counter()
    values = []
    for atoms in frames:
        # Explicit calculate avoids ASE reusing an identical frame's result.
        calc.calculate(atoms, ["energy", "forces"])
        values.append((calc.results["energy"], calc.results["forces"].copy()))
    torch.cuda.synchronize(calc.backend.device)
    return values, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chgnet")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        parser.error("This comparison requires an available CUDA GPU")
    if args.frames < 1:
        parser.error("--frames must be positive")
    atoms = read(args.input) if args.input else bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(args.frames):
        frame = atoms.copy()
        frame.positions += rng.normal(0, .01, frame.positions.shape)
        frames.append(frame)
    timings, predictions = {}, {}
    # Run serially to avoid retaining two models' graph pools in memory.
    for mode, enabled in [("eager", False), ("cuda_graph", True)]:
        calc = FastMDCalculator(args.model, checkpoint=args.checkpoint, device=args.device, cuda_graph=enabled)
        _, setup = evaluate(calc, frames)
        predictions[mode], timings[mode] = evaluate(calc, frames)
        print(f"{mode}: setup={setup:.3f}s, measured={timings[mode]*1000/len(frames):.3f} ms/frame")
        print(calc.stats())
        calc.clear_cache()
        del calc
    max_e = max_f = 0.
    for (e0, f0), (e1, f1) in zip(predictions["eager"], predictions["cuda_graph"]):
        np.testing.assert_allclose(e0, e1, atol=3e-3, rtol=1e-5)
        np.testing.assert_allclose(f0, f1, atol=3e-3, rtol=1e-3)
        max_e = max(max_e, abs(e0-e1))
        max_f = max(max_f, np.max(np.abs(f0-f1)))
    print(f"PASS: max |ΔE|={max_e:.6g} eV; max |ΔF|={max_f:.6g} eV/Å")
    print(f"Measured speedup: {timings['eager']/timings['cuda_graph']:.2f}x")


if __name__ == "__main__":
    main()
