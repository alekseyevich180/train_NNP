from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.data import vdw_radii
from ase.io import read, write
from ase.optimize import LBFGS


KCAL_MOL_TO_EV = units.kcal / units.mol


def path_candidates(path: Path, script_file: str | None = None) -> list[Path]:
    if path.is_absolute():
        return [path]

    candidates = [Path.cwd() / path]
    if script_file is not None:
        script_dir = Path(script_file).resolve().parent
        candidates.extend([script_dir / path, script_dir.parent / path])
    candidates.append(Path.cwd() / "large_time_scale" / path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def resolve_existing_path(path_text: str | Path, label: str, script_file: str | None = None) -> Path:
    path = Path(path_text)
    for candidate in path_candidates(path, script_file=script_file):
        if candidate.exists():
            return candidate
    checked = "\n".join(str(candidate) for candidate in path_candidates(path, script_file=script_file))
    raise FileNotFoundError(f"{label} not found: {path}\nChecked:\n{checked}")


def resolve_output_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


@dataclass(frozen=True)
class TDBBParameters:
    """Parameters for time-dependent bond boost bias."""

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


@dataclass(frozen=True)
class TemplateData:
    target_by_symbol_pair: dict[str, float]
    distances_by_symbol_pair: dict[str, tuple[float, ...]]
    names: tuple[str, ...]


class TDBBBias:
    """Compute TDBB bias energy and forces for selected atom pairs."""

    def __init__(
        self,
        atoms: Atoms,
        pairs: Sequence[tuple[int, int]],
        params: TDBBParameters,
        target_distances: Sequence[float] | None = None,
    ):
        self.params = params
        self.pairs = tuple(pairs)
        self.time_ps = 0.0
        self.target_distances = (
            tuple(float(distance) for distance in target_distances)
            if target_distances is not None
            else self._make_target_distances(atoms)
        )
        if len(self.target_distances) != len(self.pairs):
            raise ValueError("target_distances must have the same length as pairs.")

    def set_time(self, time_ps: float) -> None:
        self.time_ps = max(0.0, float(time_ps))

    def current_f1_ev(self) -> float:
        return min(self.params.gamma_ev_ps * self.time_ps, self.params.f1_max_ev)

    def calculate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        f1 = self.current_f1_ev()
        forces = np.zeros_like(atoms.get_positions())
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


class DeepMDWriter:
    def __init__(self, atoms: Atoms, root: Path, set_interval: int):
        self.root = root
        self.set_interval = int(set_interval)
        self.root.mkdir(parents=True, exist_ok=True)
        self.write_type_files(atoms)

    def write_type_files(self, atoms: Atoms) -> None:
        symbols = atoms.get_chemical_symbols()
        unique_symbols = sorted(set(symbols))
        type_map = {symbol: index for index, symbol in enumerate(unique_symbols)}
        type_list = [type_map[symbol] for symbol in symbols]

        np.savetxt(self.root / "type.raw", np.asarray(type_list), fmt="%d")
        with (self.root / "type_map.raw").open("w") as file:
            for symbol in unique_symbols:
                file.write(symbol + "\n")

    def add_frame(self, atoms: Atoms, step_id: int) -> None:
        set_id = step_id // self.set_interval
        set_dir = self.root / f"set_{set_id:03d}" / f"set.{step_id}"
        set_dir.mkdir(parents=True, exist_ok=True)

        np.save(set_dir / "coord.npy", np.asarray([atoms.get_positions().reshape(-1)]))
        np.save(set_dir / "force.npy", np.asarray([atoms.get_forces().reshape(-1)]))
        np.save(set_dir / "energy.npy", np.asarray([atoms.get_potential_energy()]))
        np.save(set_dir / "box.npy", np.asarray([atoms.get_cell().array.reshape(-1)]))


def get_aimd_fixed_indices(atoms: Atoms, fixed_z_lower: float, fixed_z_upper: float) -> list[int]:
    return [atom.index for atom in atoms if fixed_z_lower <= atom.position[2] <= fixed_z_upper]


def relax_surface(
    atoms: Atoms,
    fixed_z_lower: float,
    fixed_z_upper: float,
    surface_depth: float,
    fmax: float,
    logfile: str = "surface_relax.log",
) -> None:
    print("Starting surface relaxation...")
    max_z = max(atoms.positions[:, 2])
    surface_z = max_z - surface_depth

    freeze = []
    for atom in atoms:
        if atom.symbol not in ["Zn", "O"]:
            freeze.append(atom.index)
        elif fixed_z_lower <= atom.position[2] <= fixed_z_upper:
            freeze.append(atom.index)
        elif atom.position[2] < surface_z:
            freeze.append(atom.index)

    atoms.set_constraint(FixAtoms(indices=freeze))
    opt = LBFGS(atoms, logfile=logfile)
    opt.run(fmax=fmax)
    atoms.set_constraint()
    print("Surface relaxation finished")


def relax_whole_structure(
    atoms: Atoms,
    fixed_z_lower: float,
    fixed_z_upper: float,
    fmax: float,
    logfile: str = "whole_relax.log",
) -> None:
    print("Starting whole-structure relaxation...")
    fixed = get_aimd_fixed_indices(atoms, fixed_z_lower, fixed_z_upper)
    atoms.set_constraint(FixAtoms(indices=fixed))
    opt = LBFGS(atoms, logfile=logfile)
    opt.run(fmax=fmax)
    atoms.set_constraint()
    print("Whole-structure relaxation finished")


def get_pressure_info(atoms: Atoms) -> dict[str, object]:
    try:
        stress = atoms.get_stress(voigt=True)
    except Exception as exc:
        return {"stress_eV_A3": None, "pressure_GPa": None, "pressure_error": str(exc)}

    pressure_ev_a3 = -float(np.mean(stress[:3]))
    return {
        "stress_eV_A3": [float(x) for x in stress],
        "pressure_GPa": pressure_ev_a3 * 160.21766208,
        "pressure_error": None,
    }


def save_restart(
    atoms: Atoms,
    restart_root: Path,
    step_id: int,
    target_temperature: float,
    phase: str,
    timestep_fs: float,
    tau_t_fs: float,
    input_file: str,
    calc_mode: str,
) -> None:
    checkpoint_dir = restart_root / f"checkpoint_{step_id:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write(checkpoint_dir / "atoms.traj", atoms)
    write(checkpoint_dir / "atoms.cif", atoms)

    state = {
        "step": int(step_id),
        "phase": phase,
        "target_temperature_K": float(target_temperature),
        "instant_temperature_K": float(atoms.get_temperature()),
        "timestep_fs": float(timestep_fs),
        "tau_t_fs": float(tau_t_fs),
        "input_file": input_file,
        "calc_mode": calc_mode,
    }
    state.update(get_pressure_info(atoms))

    with (checkpoint_dir / "state.json").open("w") as file:
        json.dump(state, file, indent=2)
    with (restart_root / "latest_checkpoint.txt").open("w") as file:
        file.write(str(checkpoint_dir) + "\n")

    print(f"restart checkpoint saved: {checkpoint_dir}")


def load_restart(
    restart_from: str,
    calculator: Calculator,
    pbc: bool,
    fixed_z_lower: float,
    fixed_z_upper: float,
    script_file: str | None = None,
) -> tuple[Atoms, dict[str, object]]:
    restart_path = resolve_existing_path(restart_from, "Restart path", script_file=script_file)
    if restart_path.is_dir():
        checkpoint_dir = restart_path
    else:
        checkpoint_dir = resolve_existing_path(restart_path.read_text().strip(), "Checkpoint path", script_file=script_file)

    atoms = read(checkpoint_dir / "atoms.traj")
    with (checkpoint_dir / "state.json").open() as file:
        state = json.load(file)

    atoms.calc = calculator
    atoms.pbc = pbc
    fixed = get_aimd_fixed_indices(atoms, fixed_z_lower, fixed_z_upper)
    atoms.set_constraint(FixAtoms(indices=fixed))

    print(f"Restart from {checkpoint_dir}")
    print(f"Restart step = {state['step']}, target T = {state['target_temperature_K']} K")
    return atoms, state


def make_md_segments(
    initial_temp: float,
    final_temp: float,
    ramp_interval: float,
    ramp_steps: int,
    stab_steps: int,
    prod_steps: int,
    initial_steps: int = 10000,
) -> list[dict[str, float | int | str]]:
    segments: list[dict[str, float | int | str]] = [
        {"phase": "initial", "temperature": initial_temp, "steps": initial_steps}
    ]

    curr_t = initial_temp
    while curr_t < final_temp:
        curr_t += ramp_interval
        if curr_t > final_temp:
            curr_t = final_temp
        segments.append({"phase": "ramp", "temperature": curr_t, "steps": ramp_steps})
        segments.append({"phase": "stabilization", "temperature": curr_t, "steps": stab_steps})

    segments.append({"phase": "production", "temperature": final_temp, "steps": prod_steps})
    return segments


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


def parse_symbol_pair(symbol_pair_text: str) -> tuple[str, str]:
    item = symbol_pair_text.strip()
    if "=" in item:
        left, right = item.split("=", maxsplit=1)
    elif "-" in item:
        left, right = item.split("-", maxsplit=1)
    elif ":" in item:
        left, right = item.split(":", maxsplit=1)
    else:
        raise ValueError(f"Invalid symbol pair '{item}'. Use forms like C-O or C:O.")

    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"Invalid symbol pair '{item}'.")
    return left, right


def symbol_pair_key(symbol_a: str, symbol_b: str) -> str:
    return "-".join(sorted((symbol_a, symbol_b)))


def template_bond_key(symbol_pair_text: str) -> str:
    item = symbol_pair_text.strip()
    if item == "C=C":
        return "C=C"
    symbol_a, symbol_b = parse_symbol_pair(item)
    return symbol_pair_key(symbol_a, symbol_b)


def parse_template_bond_list(symbol_pairs_text: str) -> list[str]:
    keys: list[str] = []
    for item in symbol_pairs_text.split(","):
        item = item.strip()
        if item:
            keys.append(template_bond_key(item))
    if not keys:
        raise ValueError("At least one template bond symbol pair is required.")
    return keys


def make_symbol_pairs(atoms: Atoms, symbol_pair_text: str, cutoff_ang: float) -> list[tuple[int, int]]:
    symbol_a, symbol_b = parse_symbol_pair(symbol_pair_text)
    symbols = atoms.get_chemical_symbols()
    indices_a = [i for i, symbol in enumerate(symbols) if symbol == symbol_a]
    indices_b = [i for i, symbol in enumerate(symbols) if symbol == symbol_b]

    if not indices_a:
        raise ValueError(f"No atoms with symbol '{symbol_a}' were found.")
    if not indices_b:
        raise ValueError(f"No atoms with symbol '{symbol_b}' were found.")

    pairs: list[tuple[int, int]] = []
    same_symbol = symbol_a == symbol_b
    for i in indices_a:
        for j in indices_b:
            if i == j or (same_symbol and i > j):
                continue
            if float(atoms.get_distance(i, j, mic=True)) <= cutoff_ang:
                pairs.append((i, j))

    if not pairs:
        raise ValueError(
            f"No {symbol_a}-{symbol_b} pairs were found within {cutoff_ang:.3f} A. "
            "Increase pair_cutoff_ang or specify pairs manually."
        )
    return pairs


def find_bonded_indices(atoms: Atoms, center_index: int, symbol: str, cutoff_ang: float) -> list[int]:
    symbols = atoms.get_chemical_symbols()
    bonded: list[int] = []
    for index, atom_symbol in enumerate(symbols):
        if index == center_index or atom_symbol != symbol:
            continue
        if float(atoms.get_distance(center_index, index, mic=True)) <= cutoff_ang:
            bonded.append(index)
    return bonded


def find_carbon_double_bonds(atoms: Atoms, distance_range: tuple[float, float]) -> list[tuple[int, int]]:
    symbols = atoms.get_chemical_symbols()
    carbon_indices = [i for i, symbol in enumerate(symbols) if symbol == "C"]
    min_dist, max_dist = distance_range
    double_bonds: list[tuple[int, int]] = []

    for pos, i in enumerate(carbon_indices):
        for j in carbon_indices[pos + 1 :]:
            distance = float(atoms.get_distance(i, j, mic=True))
            if min_dist <= distance <= max_dist:
                double_bonds.append((i, j))

    return double_bonds


def find_functional_carbon_indices(atoms: Atoms, o_c_cutoff_ang: float, c_c_shell_cutoff_ang: float) -> set[int]:
    symbols = atoms.get_chemical_symbols()
    oxygen_indices = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    carbon_indices = [i for i, symbol in enumerate(symbols) if symbol == "C"]
    directly_attached: set[int] = set()

    for c_index in carbon_indices:
        if any(float(atoms.get_distance(c_index, o_index, mic=True)) <= o_c_cutoff_ang for o_index in oxygen_indices):
            directly_attached.add(c_index)

    functional_carbons = set(directly_attached)
    for c_index in carbon_indices:
        if c_index in functional_carbons:
            continue
        if any(float(atoms.get_distance(c_index, root_c, mic=True)) <= c_c_shell_cutoff_ang for root_c in directly_attached):
            functional_carbons.add(c_index)

    return functional_carbons


def find_functional_carbon_double_bonds(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    double_bonds = find_carbon_double_bonds(atoms, (args.double_bond_min, args.double_bond_max))
    functional_carbons = find_functional_carbon_indices(
        atoms,
        args.functional_o_c_cutoff,
        args.functional_c_c_shell_cutoff,
    )
    filtered = [pair for pair in double_bonds if pair[0] in functional_carbons or pair[1] in functional_carbons]

    print(f"Functional-near C atoms: {sorted(functional_carbons)}")
    if filtered:
        return filtered

    raise ValueError(
        "C=C bonds were found, but none were near the detected functional-group carbons. "
        "Increase functional_o_c_cutoff/functional_c_c_shell_cutoff or use manual pairs."
    )


def find_enol_like_oxygen_indices(
    atoms: Atoms,
    double_bonds: Sequence[tuple[int, int]],
    o_h_cutoff_ang: float,
    c_o_cutoff_ang: float,
) -> list[int]:
    symbols = atoms.get_chemical_symbols()
    double_bond_carbons = {index for pair in double_bonds for index in pair}
    oxygen_indices: list[int] = []

    for index, symbol in enumerate(symbols):
        if symbol != "O":
            continue
        bonded_h = find_bonded_indices(atoms, index, "H", o_h_cutoff_ang)
        bonded_double_bond_c = [
            c_index
            for c_index in double_bond_carbons
            if float(atoms.get_distance(index, c_index, mic=True)) <= c_o_cutoff_ang
        ]
        if bonded_h or bonded_double_bond_c:
            oxygen_indices.append(index)

    return oxygen_indices


def make_double_bond_pairs(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    double_bonds = find_functional_carbon_double_bonds(atoms, args)
    if not double_bonds:
        raise ValueError(
            "No C=C double bonds were found. Adjust double_bond_min/double_bond_max "
            "or use --pair-mode manual."
        )
    print(f"Detected C=C bonds: {double_bonds}")
    return double_bonds


def make_double_bond_c_o_pairs(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    double_bonds = make_double_bond_pairs(atoms, args)
    oxygen_indices = find_enol_like_oxygen_indices(
        atoms,
        double_bonds,
        args.o_h_cutoff,
        args.enol_c_o_cutoff,
    )
    if not oxygen_indices:
        raise ValueError(
            "No enol-like oxygen atoms were found. Adjust o_h_cutoff/enol_c_o_cutoff "
            "or use --pair-mode symbols/manual."
        )

    pairs: list[tuple[int, int]] = []
    double_bond_carbons = sorted({index for pair in double_bonds for index in pair})
    for c_index in double_bond_carbons:
        for o_index in oxygen_indices:
            c_o_distance = float(atoms.get_distance(c_index, o_index, mic=True))
            if c_o_distance <= args.existing_c_o_cutoff:
                continue
            if c_o_distance <= args.pair_cutoff:
                pairs.append((c_index, o_index))

    if not pairs:
        raise ValueError(
            "C=C carbons and enol-like oxygens were found, but no new C-O target pairs "
            f"were within {args.pair_cutoff:.3f} A. Increase pair_cutoff or use manual pairs."
        )

    print(f"Detected C=C bonds: {double_bonds}")
    print(f"Detected enol-like O atoms: {oxygen_indices}")
    return pairs


def make_double_bond_and_c_o_pairs(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    pairs = []
    seen: set[tuple[int, int]] = set()
    for pair in [*make_double_bond_pairs(atoms, args), *make_double_bond_c_o_pairs(atoms, args)]:
        ordered = tuple(sorted(pair))
        if ordered not in seen:
            seen.add(ordered)
            pairs.append(pair)
    return pairs


def make_functional_c_c_pairs(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    functional_carbons = sorted(
        find_functional_carbon_indices(
            atoms,
            args.functional_o_c_cutoff,
            args.functional_c_c_shell_cutoff,
        )
    )
    pairs: list[tuple[int, int]] = []
    for pos, i in enumerate(functional_carbons):
        for j in functional_carbons[pos + 1 :]:
            if float(atoms.get_distance(i, j, mic=True)) <= args.functional_c_c_pair_cutoff:
                pairs.append((i, j))

    if not pairs:
        raise ValueError(
            "No functional-near C-C candidate pairs were found. Increase "
            "functional_c_c_pair_cutoff or specify pairs manually."
        )

    print(f"Functional-near C atoms for C=C formation: {functional_carbons}")
    print(f"Functional-near C-C candidates: {pairs}")
    return pairs


def make_functional_c_o_pairs(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    symbols = atoms.get_chemical_symbols()
    functional_carbons = sorted(
        find_functional_carbon_indices(
            atoms,
            args.functional_o_c_cutoff,
            args.functional_c_c_shell_cutoff,
        )
    )
    oxygen_indices = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    pairs: list[tuple[int, int]] = []

    for c_index in functional_carbons:
        for o_index in oxygen_indices:
            distance = float(atoms.get_distance(c_index, o_index, mic=True))
            if distance <= args.existing_c_o_cutoff:
                continue
            if distance <= args.pair_cutoff:
                pairs.append((c_index, o_index))

    if pairs:
        print(f"Functional-near C-O candidates: {pairs}")
    else:
        print("Warning: no functional-near C-O candidates were found; continuing with functional-near C-C candidates only.")
    return pairs


def make_functional_c_c_and_c_o_pairs(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    pairs = []
    seen: set[tuple[int, int]] = set()
    for pair in [*make_functional_c_c_pairs(atoms, args), *make_functional_c_o_pairs(atoms, args)]:
        ordered = tuple(sorted(pair))
        if ordered not in seen:
            seen.add(ordered)
            pairs.append(pair)
    return pairs


def list_template_files(template_dir: Path, script_file: str | None = None) -> list[Path]:
    template_dir = resolve_existing_path(template_dir, "Template directory", script_file=script_file)
    if not template_dir.is_dir():
        raise NotADirectoryError(f"Template path is not a directory: {template_dir}")

    suffixes = {".cif", ".xyz", ".traj", ".vasp", ".poscar", ".pdb"}
    files = [path for path in sorted(template_dir.iterdir()) if path.is_file() and path.suffix.lower() in suffixes]
    if not files:
        raise FileNotFoundError(f"No supported template structure files were found in {template_dir}")
    return files


def template_distances_for_bond_key(
    atoms: Atoms,
    bond_key: str,
    cutoff_ang: float,
    c_c_double_range: tuple[float, float],
    c_c_single_range: tuple[float, float],
) -> list[float]:
    symbols = atoms.get_chemical_symbols()
    distances: list[float] = []

    if bond_key == "C=C":
        symbol_a = symbol_b = "C"
        min_dist, max_dist = c_c_double_range
    elif bond_key == "C-C":
        symbol_a = symbol_b = "C"
        min_dist, max_dist = c_c_single_range
    else:
        symbol_a, symbol_b = parse_symbol_pair(bond_key)
        min_dist, max_dist = 0.2, cutoff_ang

    indices_a = [i for i, symbol in enumerate(symbols) if symbol == symbol_a]
    indices_b = [i for i, symbol in enumerate(symbols) if symbol == symbol_b]
    same_symbol = symbol_a == symbol_b

    for i in indices_a:
        for j in indices_b:
            if i == j or (same_symbol and i > j):
                continue
            distance = float(atoms.get_distance(i, j, mic=True))
            if min_dist <= distance <= max_dist:
                distances.append(distance)

    return distances


def load_template_data(args: argparse.Namespace, script_file: str | None = None) -> TemplateData:
    bond_keys = parse_template_bond_list(args.template_bond_symbols)
    files = list_template_files(Path(args.template_dir), script_file=script_file)
    names: list[str] = []
    distances_by_key: dict[str, list[float]] = {key: [] for key in bond_keys}

    for file_path in files:
        template_atoms = read(file_path)
        names.append(file_path.stem)
        for key in bond_keys:
            distances_by_key[key].extend(
                template_distances_for_bond_key(
                    template_atoms,
                    key,
                    args.template_bond_cutoff,
                    (args.template_double_min, args.template_double_max),
                    (args.template_single_min, args.template_single_max),
                )
            )

    target_by_key: dict[str, float] = {}
    frozen_distances_by_key: dict[str, tuple[float, ...]] = {}
    for key, distances in distances_by_key.items():
        if not distances:
            raise ValueError(
                f"No template {key} distances within {args.template_bond_cutoff:.3f} A "
                f"were found in {args.template_dir}."
            )
        arr = np.asarray(distances, dtype=float)
        frozen_distances_by_key[key] = tuple(float(x) for x in arr)
        if args.template_target_mode == "min":
            target_by_key[key] = float(np.min(arr))
        elif args.template_target_mode == "mean":
            target_by_key[key] = float(np.mean(arr))
        else:
            raise ValueError("template_target_mode must be 'min' or 'mean'.")

    print(f"Loaded template structures: {names}")
    print(f"Template target distances by bond type (A): {target_by_key}")
    if args.template_pair_source in {"functional-cc", "functional-cc-and-co"} and "C=C" not in target_by_key:
        raise ValueError("functional-cc template mode requires a C=C target in template_bond_symbols/interm structures.")

    return TemplateData(
        target_by_symbol_pair=target_by_key,
        distances_by_symbol_pair=frozen_distances_by_key,
        names=tuple(names),
    )


def current_pair_distances(atoms: Atoms, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray([float(atoms.get_distance(i, j, mic=True)) for i, j in pairs], dtype=float)


def get_template_target_key(
    atoms: Atoms,
    pair: tuple[int, int],
    template_data: TemplateData,
    args: argparse.Namespace,
) -> str:
    symbols = atoms.get_chemical_symbols()
    i, j = pair
    key = symbol_pair_key(symbols[i], symbols[j])
    if key == "C-C":
        distance = float(atoms.get_distance(i, j, mic=True))
        if (
            args.template_pair_source in {"functional-cc", "functional-cc-and-co"}
            and "C=C" in template_data.target_by_symbol_pair
        ):
            key = "C=C"
        elif args.template_double_min <= distance <= args.template_double_max and "C=C" in template_data.target_by_symbol_pair:
            key = "C=C"
    if key not in template_data.target_by_symbol_pair:
        raise ValueError(
            f"No template target was loaded for pair {i}-{j} ({key}). "
            "Add this bond type to template_bond_symbols or remove the pair."
        )
    return key


def get_template_target_distances(
    atoms: Atoms,
    pairs: Sequence[tuple[int, int]],
    template_data: TemplateData,
    args: argparse.Namespace,
) -> tuple[float, ...]:
    return tuple(
        template_data.target_by_symbol_pair[get_template_target_key(atoms, pair, template_data, args)]
        for pair in pairs
    )


def get_reactive_pairs(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.pair_mode == "manual":
        return parse_pairs(args.pairs)
    if args.pair_mode == "symbols":
        return make_symbol_pairs(atoms, args.symbols, args.pair_cutoff)
    if args.pair_mode == "double-bond-co":
        return make_double_bond_c_o_pairs(atoms, args)
    if args.pair_mode == "template":
        return get_template_source_pairs(atoms, args)
    raise ValueError("pair_mode must be 'manual', 'symbols', 'double-bond-co', or 'template'.")


def get_template_source_pairs(atoms: Atoms, args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.template_pair_source in {"functional-cc", "functional-cc-and-co"}:
        print("Template source mode forms new C=C targets from functional-near C-C candidates.")
    if args.template_pair_source == "manual":
        return parse_pairs(args.pairs)
    if args.template_pair_source == "symbols":
        return make_symbol_pairs(atoms, args.symbols, args.pair_cutoff)
    if args.template_pair_source == "double-bond-co":
        return make_double_bond_c_o_pairs(atoms, args)
    if args.template_pair_source == "double-bond-and-co":
        return make_double_bond_and_c_o_pairs(atoms, args)
    if args.template_pair_source == "functional-cc":
        return make_functional_c_c_pairs(atoms, args)
    if args.template_pair_source == "functional-cc-and-co":
        return make_functional_c_c_and_c_o_pairs(atoms, args)
    raise ValueError(
        "template_pair_source must be 'manual', 'symbols', 'double-bond-co', "
        "'double-bond-and-co', 'functional-cc', or 'functional-cc-and-co'."
    )

