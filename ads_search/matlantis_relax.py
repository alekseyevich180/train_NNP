from pathlib import Path
import argparse

from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import LBFGS
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode


# Edit this block directly if you prefer not to pass command-line arguments.
CONFIG = {
    "system": {
        "input_file": "structure.cif",
        "output_file": "relaxed.cif",
        "trajectory_file": "relax.traj",
        "log_file": "relax.log",
        "pbc": True,
    },
    "relaxation": {
        "fmax": 0.05,
        "steps": 500,
    },
    "constraints": {
        "fix_below_z": None,
        "fix_symbols": None,
    },
}


def build_calculator() -> ASECalculator:
    estimator = Estimator(calc_mode=EstimatorCalcMode.PBE_U_PLUS_D3)
    return ASECalculator(estimator)


def get_fixed_indices(atoms, z_max: float | None, symbols: list[str] | None) -> list[int]:
    fixed = set()

    if z_max is not None:
        fixed.update(atom.index for atom in atoms if atom.position[2] <= z_max)

    if symbols:
        symbol_set = set(symbols)
        fixed.update(atom.index for atom in atoms if atom.symbol in symbol_set)

    return sorted(fixed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structure relaxation with Matlantis calculator.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(CONFIG["system"]["input_file"]),
        help="Input structure file readable by ASE.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CONFIG["system"]["output_file"]),
        help="Optimized structure path.",
    )
    parser.add_argument(
        "--traj",
        type=Path,
        default=Path(CONFIG["system"]["trajectory_file"]),
        help="Optimization trajectory path.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(CONFIG["system"]["log_file"]),
        help="Optimizer log path.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=CONFIG["relaxation"]["fmax"],
        help="Force convergence threshold in eV/A.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=CONFIG["relaxation"]["steps"],
        help="Maximum optimization steps.",
    )
    parser.add_argument(
        "--fix-below-z",
        type=float,
        default=CONFIG["constraints"]["fix_below_z"],
        help="Fix atoms with z <= this value.",
    )
    parser.add_argument(
        "--fix-symbols",
        nargs="+",
        default=CONFIG["constraints"]["fix_symbols"],
        help="Fix atoms whose element symbols match these values.",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring unknown arguments: {' '.join(unknown_args)}")

    atoms = read(args.input)
    atoms.calc = build_calculator()
    atoms.pbc = CONFIG["system"]["pbc"]

    fixed_indices = get_fixed_indices(atoms, args.fix_below_z, args.fix_symbols)
    if fixed_indices:
        atoms.set_constraint(FixAtoms(indices=fixed_indices))
        print(f"Fixed {len(fixed_indices)} atoms")
    else:
        print("No atomic constraints applied")

    print(f"Initial energy: {atoms.get_potential_energy():.6f} eV")

    optimizer = LBFGS(atoms, trajectory=str(args.traj), logfile=str(args.log))
    optimizer.run(fmax=args.fmax, steps=args.steps)

    final_energy = atoms.get_potential_energy()
    print(f"Final energy: {final_energy:.6f} eV")
    write(args.output, atoms)
    print(f"Relaxed structure written to {args.output}")


if __name__ == "__main__":
    main()
