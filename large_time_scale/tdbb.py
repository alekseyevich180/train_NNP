from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.data import vdw_radii
from ase.io import Trajectory, read
from ase.md import MDLogger
from ase.md.langevin import Langevin
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode


KCAL_MOL_TO_EV = units.kcal / units.mol


@dataclass(frozen=True)
class TDBBParameters:
    """Parameters for time-dependent bond boost bias.

    gamma grows the maximum bias height linearly with simulation time:
    f1(t) = min(gamma * t_ps, f1_max).
    """

    gamma_kcal_mol_ps: float = 1.0
    f1_max_kcal_mol: float = 250.0
    f2_inv_ang2: float = 10.0
    target_scale: float = 0.60
    default_target_ang: float = 1.5

    @property
    def gamma_ev_ps(self) -> float:
        return self.gamma_kcal_mol_ps * KCAL_MOL_TO_EV

    @property
    def f1_max_ev(self) -> float:
        return self.f1_max_kcal_mol * KCAL_MOL_TO_EV


class TDBBBias:
    """Compute TDBB bias energy and forces for selected atom pairs."""

    def __init__(self, atoms: Atoms, pairs: Sequence[tuple[int, int]], params: TDBBParameters):
        self.params = params
        self.pairs = tuple(pairs)
        self.time_ps = 0.0
        self.target_distances = self._make_target_distances(atoms)

    def set_time(self, time_ps: float) -> None:
        self.time_ps = max(0.0, float(time_ps))

    def current_f1_ev(self) -> float:
        return min(self.params.gamma_ev_ps * self.time_ps, self.params.f1_max_ev)

    def calculate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Return bias energy in eV and bias forces in eV/A."""
        f1 = self.current_f1_ev()
        positions = atoms.get_positions()
        forces = np.zeros_like(positions)
        energy = 0.0

        if f1 <= 0.0:
            return energy, forces

        for pair_index, (i, j) in enumerate(self.pairs):
            diff_vec = atoms.get_distance(i, j, vector=True, mic=True)
            distance = float(np.linalg.norm(diff_vec))
            if distance <= 1.0e-12:
                continue

            target = self.target_distances[pair_index]
            delta = distance - target
            if delta <= 0.0:
                continue

            exp_term = float(np.exp(-self.params.f2_inv_ang2 * delta**2))
            energy += f1 * (1.0 - exp_term)

            force_mag = 2.0 * f1 * self.params.f2_inv_ang2 * delta * exp_term
            unit_vec = diff_vec / distance

            # ASE's distance vector points from atom i to atom j. Positive force on i
            # therefore pulls i toward j and shortens an overlong reactive pair.
            forces[i] += force_mag * unit_vec
            forces[j] -= force_mag * unit_vec

        return energy, forces

    def _make_target_distances(self, atoms: Atoms) -> tuple[float, ...]:
        targets: list[float] = []
        symbols = atoms.get_chemical_symbols()
        numbers = atoms.get_atomic_numbers()
        natoms = len(atoms)

        for i, j in self.pairs:
            if i == j:
                raise ValueError(f"Reactive pair ({i}, {j}) uses the same atom twice.")
            if not (0 <= i < natoms and 0 <= j < natoms):
                raise ValueError(f"Reactive pair ({i}, {j}) is outside atom index range 0-{natoms - 1}.")

            ri = float(vdw_radii[numbers[i]])
            rj = float(vdw_radii[numbers[j]])
            if np.isfinite(ri) and np.isfinite(rj) and ri > 0.0 and rj > 0.0:
                targets.append(self.params.target_scale * (ri + rj))
            else:
                targets.append(self.params.default_target_ang)
                print(
                    f"Warning: missing vdW radius for {symbols[i]}-{symbols[j]}; "
                    f"using {self.params.default_target_ang:.3f} A."
                )

        return tuple(targets)


class BiasedCalculator(Calculator):
    """ASE calculator wrapper that adds TDBB bias to a base calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calculator: Calculator, bias: TDBBBias):
        super().__init__()
        self.base_calculator = base_calculator
        self.bias = bias

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy", "forces"),
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("BiasedCalculator requires an Atoms object.")

        base_energy = float(self.base_calculator.get_potential_energy(atoms))
        base_forces = np.asarray(self.base_calculator.get_forces(atoms), dtype=float)
        bias_energy, bias_forces = self.bias.calculate(atoms)

        self.results["energy"] = base_energy + bias_energy
        self.results["forces"] = base_forces + bias_forces


def parse_pairs(pair_text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in pair_text.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            left, right = item.split("-", maxsplit=1)
        elif ":" in item:
            left, right = item.split(":", maxsplit=1)
        else:
            raise ValueError(f"Invalid pair '{item}'. Use forms like 0-10,1-8.")

        pairs.append((int(left), int(right)))

    if not pairs:
        raise ValueError("At least one reactive pair is required.")
    return pairs


def parse_calc_mode(calc_mode_name: str) -> EstimatorCalcMode:
    try:
        return EstimatorCalcMode[calc_mode_name]
    except KeyError as exc:
        valid_names = ", ".join(mode.name for mode in EstimatorCalcMode)
        raise ValueError(f"Unknown calc mode '{calc_mode_name}'. Valid modes: {valid_names}") from exc


def build_pfp_calculator(calc_mode_name: str) -> Calculator:
    estimator = Estimator(calc_mode=parse_calc_mode(calc_mode_name))
    return ASECalculator(estimator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TDBB accelerated Langevin MD using PFP as the base potential."
    )
    parser.add_argument("-i", "--input", default="monomers.xyz", help="Initial structure file.")
    parser.add_argument("-o", "--trajectory", default="accelerated_md.traj", help="Output ASE trajectory.")
    parser.add_argument("--log", default="accelerated_md.log", help="MD log file.")
    parser.add_argument("--pairs", default="0-10", help="Reactive atom pairs, e.g. '0-10,4-15'.")
    parser.add_argument("--steps", type=int, default=4000, help="Number of MD steps.")
    parser.add_argument("--timestep-fs", type=float, default=0.25, help="MD time step in fs.")
    parser.add_argument("--temperature-k", type=float, default=300.0, help="Langevin temperature in K.")
    parser.add_argument("--friction", type=float, default=0.01, help="ASE Langevin friction parameter.")
    parser.add_argument("--traj-interval", type=int, default=100, help="Trajectory write interval.")
    parser.add_argument("--log-interval", type=int, default=100, help="MD log interval.")
    parser.add_argument("--gamma", type=float, default=1.0, help="TDBB gamma in kcal/(mol ps).")
    parser.add_argument("--f1-max", type=float, default=250.0, help="Maximum TDBB bias depth in kcal/mol.")
    parser.add_argument("--f2", type=float, default=10.0, help="TDBB range parameter in A^-2.")
    parser.add_argument("--target-scale", type=float, default=0.60, help="Target distance scale for vdW radii sum.")
    parser.add_argument("--default-target", type=float, default=1.5, help="Fallback target distance in A.")
    parser.add_argument(
        "--calc-mode",
        default="PBE_U_PLUS_D3",
        help="EstimatorCalcMode name, e.g. PBE_U_PLUS_D3.",
    )
    return parser.parse_args()


def run_simulation(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input structure not found: {input_path}")

    pairs = parse_pairs(args.pairs)
    params = TDBBParameters(
        gamma_kcal_mol_ps=args.gamma,
        f1_max_kcal_mol=args.f1_max,
        f2_inv_ang2=args.f2,
        target_scale=args.target_scale,
        default_target_ang=args.default_target,
    )

    atoms = read(input_path)
    bias = TDBBBias(atoms, pairs, params)
    atoms.calc = BiasedCalculator(build_pfp_calculator(args.calc_mode), bias)

    dyn = Langevin(
        atoms,
        args.timestep_fs * units.fs,
        temperature_K=args.temperature_k,
        friction=args.friction,
    )

    def update_bias_time() -> None:
        bias.set_time(dyn.get_number_of_steps() * args.timestep_fs / 1000.0)
        if atoms.calc is not None:
            atoms.calc.reset()

    dyn.attach(update_bias_time, interval=1)
    dyn.attach(Trajectory(args.trajectory, "w", atoms).write, interval=args.traj_interval)
    dyn.attach(MDLogger(dyn, atoms, args.log, header=True, stress=False, peratom=False), interval=args.log_interval)

    print("Starting TDBB accelerated MD")
    print(f"Input: {input_path}")
    print(f"Reactive pairs: {pairs}")
    print(f"Target distances (A): {[round(x, 3) for x in bias.target_distances]}")
    print(f"PFP calc mode: {args.calc_mode}")
    print(f"Steps: {args.steps}, timestep: {args.timestep_fs} fs, temperature: {args.temperature_k} K")
    print(f"Trajectory: {args.trajectory}")
    print(f"Log: {args.log}")

    dyn.run(args.steps)


def main() -> None:
    run_simulation(parse_args())


if __name__ == "__main__":
    main()
