"""Run after installing fastMD: python examples/single_point.py --model chgnet."""
import argparse
from ase.build import bulk
from ase.io import read
from fastmd import FastMDCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chgnet")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", help="Any structure file readable by ASE")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    atoms = read(args.input) if args.input else bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.calc = FastMDCalculator(args.model, checkpoint=args.checkpoint, device=args.device)
    print("Energy [eV]:", atoms.get_potential_energy())
    print("Forces [eV/Å]:\n", atoms.get_forces())
    print(atoms.calc.stats())


if __name__ == "__main__":
    main()
