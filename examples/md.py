"""Standard ASE NVT MD; model inference selects CUDA Graph automatically."""
import argparse
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from fastmd import FastMDCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chgnet")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    atoms.calc = FastMDCalculator(args.model, checkpoint=args.checkpoint, device=args.device)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))
    atoms.calc.warmup(atoms)
    with Langevin(atoms, timestep=units.fs, temperature_K=300, friction=.01/units.fs,
                  trajectory="md.traj", logfile="md.log", loginterval=10) as dynamics:
        dynamics.run(args.steps)
    print(atoms.calc.stats())


if __name__ == "__main__":
    main()
