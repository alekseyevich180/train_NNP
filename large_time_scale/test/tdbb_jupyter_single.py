from __future__ import annotations

import sys

# ============================================================
# Inlined from large_time_scale/share/bonds.py
# ============================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


def parse_symbol_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def pair_key(symbol_a: str, symbol_b: str) -> str:
    return "-".join(sorted((symbol_a, symbol_b)))


@dataclass(frozen=True)
class InterfaceBondConfig:
    enabled: bool = True
    molecule_seed_symbols: tuple[str, ...] = ("C", "H")
    molecule_symbols: tuple[str, ...] = ("C", "H", "O", "N", "S")
    molecule_bond_symbols: tuple[str, ...] = ("C",)
    surface_symbols: tuple[str, ...] = ("Zn", "O")
    bond_cutoff_scale: float = 1.25
    min_bond_cutoff_ang: float = 0.7
    max_bond_cutoff_ang: float = 2.4
    min_bonds_to_stabilize: int = 2
    detection_interval: int = 10
    stable_steps: int = 20000
    restraint_k_ev_a2: float = 5.0
    progress_name: str = "interface_bonds.csv"
    event_name: str = "interface_bond_events.csv"


@dataclass(frozen=True)
class InterfaceBond:
    molecule_index: int
    surface_index: int
    distance_ang: float
    symbol_pair: str

    @property
    def pair(self) -> tuple[int, int]:
        return self.molecule_index, self.surface_index


class InterfaceBondStabilizer:
    """Temporarily restrain newly formed molecule-surface bonds."""

    def __init__(self, k_ev_a2: float, hold_steps: int):
        self.k_ev_a2 = float(k_ev_a2)
        self.hold_steps = int(hold_steps)
        self.active_until_step = -1
        self.target_distances: dict[tuple[int, int], float] = {}

    def is_active(self, step: int) -> bool:
        return step < self.active_until_step and bool(self.target_distances)

    def clear_if_expired(self, step: int) -> bool:
        if self.target_distances and step >= self.active_until_step:
            self.target_distances = {}
            return True
        return False

    def update(self, atoms: Atoms, step: int, bonds: Sequence[InterfaceBond], min_bonds: int) -> bool:
        if len(bonds) < min_bonds:
            return False
        if self.is_active(step):
            return False

        self.target_distances = {
            bond.pair: float(atoms.get_distance(*bond.pair, mic=True))
            for bond in bonds
        }
        self.active_until_step = step + self.hold_steps
        return True

    def calculate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        forces = np.zeros_like(atoms.get_positions())
        energy = 0.0

        for (i, j), target in self.target_distances.items():
            diff_vec = atoms.get_distance(i, j, vector=True, mic=True)
            distance = float(np.linalg.norm(diff_vec))
            if distance <= 1.0e-12:
                continue

            delta = distance - target
            energy += 0.5 * self.k_ev_a2 * delta**2
            force_mag = self.k_ev_a2 * delta
            unit_vec = diff_vec / distance
            forces[i] += force_mag * unit_vec
            forces[j] -= force_mag * unit_vec

        return energy, forces


class CombinedBias:
    """Sum multiple bias-like objects with calculate(atoms)."""

    def __init__(self, biases: Iterable[object]):
        self.biases = tuple(biases)

    def calculate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        total_energy = 0.0
        total_forces = np.zeros_like(atoms.get_positions())
        for bias in self.biases:
            energy, forces = bias.calculate(atoms)
            total_energy += float(energy)
            total_forces += forces
        return total_energy, total_forces


def covalent_cutoff(atoms: Atoms, i: int, j: int, config: InterfaceBondConfig) -> float:
    numbers = atoms.get_atomic_numbers()
    ri = float(covalent_radii[numbers[i]])
    rj = float(covalent_radii[numbers[j]])
    if not np.isfinite(ri) or not np.isfinite(rj) or ri <= 0.0 or rj <= 0.0:
        return config.max_bond_cutoff_ang
    cutoff = config.bond_cutoff_scale * (ri + rj)
    return min(max(cutoff, config.min_bond_cutoff_ang), config.max_bond_cutoff_ang)


def detect_molecule_indices(atoms: Atoms, config: InterfaceBondConfig) -> set[int]:
    symbols = atoms.get_chemical_symbols()
    allowed = set(config.molecule_symbols)
    seeds = {
        index
        for index, symbol in enumerate(symbols)
        if symbol in config.molecule_seed_symbols and symbol in allowed
    }
    if not seeds:
        raise ValueError(
            "No molecule seed atoms were found. Adjust molecule_seed_symbols "
            "or pass explicit molecular indices in a future extension."
        )

    molecule = set(seeds)
    changed = True
    while changed:
        changed = False
        for i, symbol_i in enumerate(symbols):
            if i in molecule or symbol_i not in allowed:
                continue
            for j in tuple(molecule):
                if float(atoms.get_distance(i, j, mic=True)) <= covalent_cutoff(atoms, i, j, config):
                    molecule.add(i)
                    changed = True
                    break

    return molecule


def detect_interface_bonds(atoms: Atoms, config: InterfaceBondConfig) -> list[InterfaceBond]:
    molecule_indices = detect_molecule_indices(atoms, config)
    symbols = atoms.get_chemical_symbols()
    molecule_bond_symbols = set(config.molecule_bond_symbols)
    surface_indices = [
        index
        for index, symbol in enumerate(symbols)
        if index not in molecule_indices and symbol in config.surface_symbols
    ]

    bonds: list[InterfaceBond] = []
    for mol_index in sorted(molecule_indices):
        if symbols[mol_index] not in molecule_bond_symbols:
            continue
        for surf_index in surface_indices:
            distance = float(atoms.get_distance(mol_index, surf_index, mic=True))
            if distance <= covalent_cutoff(atoms, mol_index, surf_index, config):
                bonds.append(
                    InterfaceBond(
                        molecule_index=mol_index,
                        surface_index=surf_index,
                        distance_ang=distance,
                        symbol_pair=pair_key(symbols[mol_index], symbols[surf_index]),
                    )
                )

    return sorted(bonds, key=lambda bond: (bond.molecule_index, bond.surface_index))


def write_interface_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "step,time_ps,bond_count,stabilizer_active,active_until_step,bonds\n"
    )


def write_interface_event_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "step,time_ps,target_temperature_K,instant_temperature_K,potential_energy_eV,bond_count,bonds\n"
    )


def append_interface_progress(
    path: Path,
    step: int,
    time_ps: float,
    bonds: Sequence[InterfaceBond],
    stabilizer: InterfaceBondStabilizer,
) -> None:
    bond_text = ";".join(
        f"{bond.molecule_index}-{bond.surface_index}:{bond.symbol_pair}:{bond.distance_ang:.4f}"
        for bond in bonds
    )
    active = stabilizer.is_active(step)
    row = [
        str(step),
        f"{time_ps:.6f}",
        str(len(bonds)),
        str(int(active)),
        str(stabilizer.active_until_step if active else ""),
        bond_text,
    ]
    with path.open("a") as file:
        file.write(",".join(row) + "\n")


def append_interface_event(
    path: Path,
    step: int,
    time_ps: float,
    target_temperature: float,
    instant_temperature: float,
    potential_energy: float,
    bonds: Sequence[InterfaceBond],
) -> None:
    bond_text = ";".join(
        f"{bond.molecule_index}-{bond.surface_index}:{bond.symbol_pair}:{bond.distance_ang:.4f}"
        for bond in bonds
    )
    row = [
        str(step),
        f"{time_ps:.6f}",
        f"{target_temperature:.6f}",
        f"{instant_temperature:.6f}",
        f"{potential_energy:.10f}",
        str(len(bonds)),
        bond_text,
    ]
    with path.open("a") as file:
        file.write(",".join(row) + "\n")

# ============================================================
# Inlined from large_time_scale/share/pfp.py
# ============================================================
from ase.calculators.calculator import Calculator
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode


def parse_calc_mode(calc_mode_name: str) -> EstimatorCalcMode:
    try:
        return EstimatorCalcMode[calc_mode_name]
    except KeyError as exc:
        valid_names = ", ".join(mode.name for mode in EstimatorCalcMode)
        raise ValueError(f"Unknown calc mode '{calc_mode_name}'. Valid modes: {valid_names}") from exc


def build_pfp_calculator(calc_mode_name: str) -> Calculator:
    estimator = Estimator(calc_mode=parse_calc_mode(calc_mode_name))
    return ASECalculator(estimator)

# ============================================================
# Inlined from large_time_scale/share/tdbb.py
# ============================================================
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

# ============================================================
# Inlined from large_time_scale/main.py
# ============================================================
import argparse
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs


SCRIPT_FILE = globals().get("__file__")


CONFIG = {
    "system": {
        "input_file": "ketone.cif",
        "fixed_z_lower_bound": 4.0,
        "fixed_z_upper_bound": 9.0,
        "surface_relax_depth": 12.0,
        "output_root": "acid_AIMD_dataset_test",
        "restart_from": None,
        "pbc": True,
        "pair_mode": "template",
        "reactive_pairs": "0-10",
        "reactive_symbols": "C-O",
        "pair_cutoff_ang": 4.0,
        "template_dir": "interm",
        "template_bond_symbols": "C-O,C=C,C-C",
        "template_bond_cutoff_ang": 2.2,
        "template_c_c_double_range_ang": (1.15, 1.45),
        "template_c_c_single_range_ang": (1.45, 1.70),
        "functional_o_c_cutoff_ang": 1.75,
        "functional_c_c_shell_cutoff_ang": 1.75,
        "functional_c_c_pair_cutoff_ang": 4.0,
        "template_pair_source": "functional-cc-and-co",
        "template_target_mode": "min",
        "double_bond_c_c_range_ang": (1.15, 1.45),
        "existing_c_o_bond_cutoff_ang": 1.65,
        "o_h_bond_cutoff_ang": 1.20,
        "enol_c_o_bond_cutoff_ang": 1.65,
    },
    "relaxation": {
        "surface_fmax": 0.05,
        "whole_fmax": 0.05,
    },
    "md_control": {
        "initial_temp": 280,
        "final_temp": 1080,
        "ramp_interval": 100,
        "ramp_steps": 20000,
        "stab_steps": 10000,
        "prod_steps": 4000000,
        "timestep": 0.5,
        "tau_t": 100.0,
    },
    "pfp": {
        "calc_mode": "PBE_U_PLUS_D3",
    },
    "tdbb": {
        "gamma": 1.0,
        "f1_max": 250.0,
        "f2": 10.0,
        "target_scale": 0.60,
        "default_target": 1.5,
    },
    "interface_bonds": {
        "enabled": True,
        "molecule_seed_symbols": "C,H",
        "molecule_symbols": "C,H,O,N,S",
        "molecule_bond_symbols": "C",
        "surface_symbols": "Zn,O",
        "bond_cutoff_scale": 1.25,
        "min_bond_cutoff_ang": 0.7,
        "max_bond_cutoff_ang": 2.4,
        "min_bonds_to_stabilize": 2,
        "detection_interval": 10,
        "stable_steps": 20000,
        "restraint_k_ev_a2": 5.0,
        "event_name": "interface_bond_events.csv",
    },
    "output": {
        "save_interval": 100,
        "deepmd_dir": "deepmd_dataset",
        "cif_dir": "cif_frames",
        "restart_dir": "restart_checkpoints",
        "deepmd_set_interval": 100000,
        "cif_set_interval": 100000,
        "restart_interval": 200000,
        "template_progress": "template_progress.csv",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learning/test entry point for the TDBB AIMD workflow using shared helpers."
    )
    parser.add_argument("-i", "--input", default=CONFIG["system"]["input_file"], help="Initial structure file.")
    parser.add_argument("--output-root", default=CONFIG["system"]["output_root"], help="Root directory for outputs.")
    parser.add_argument("--restart-from", default=CONFIG["system"]["restart_from"], help="Checkpoint directory or latest_checkpoint.txt.")
    parser.add_argument("--template-progress", default=CONFIG["output"]["template_progress"], help="Template progress CSV name.")
    parser.add_argument("--pair-mode", choices=["manual", "symbols", "double-bond-co", "template"], default=CONFIG["system"]["pair_mode"])
    parser.add_argument("--pairs", default=CONFIG["system"]["reactive_pairs"])
    parser.add_argument("--symbols", default=CONFIG["system"]["reactive_symbols"])
    parser.add_argument("--pair-cutoff", type=float, default=CONFIG["system"]["pair_cutoff_ang"])
    parser.add_argument("--template-dir", default=CONFIG["system"]["template_dir"])
    parser.add_argument("--template-bond-symbols", default=CONFIG["system"]["template_bond_symbols"])
    parser.add_argument("--template-bond-cutoff", type=float, default=CONFIG["system"]["template_bond_cutoff_ang"])
    parser.add_argument("--template-double-min", type=float, default=CONFIG["system"]["template_c_c_double_range_ang"][0])
    parser.add_argument("--template-double-max", type=float, default=CONFIG["system"]["template_c_c_double_range_ang"][1])
    parser.add_argument("--template-single-min", type=float, default=CONFIG["system"]["template_c_c_single_range_ang"][0])
    parser.add_argument("--template-single-max", type=float, default=CONFIG["system"]["template_c_c_single_range_ang"][1])
    parser.add_argument(
        "--template-pair-source",
        choices=["manual", "symbols", "double-bond-co", "double-bond-and-co", "functional-cc", "functional-cc-and-co"],
        default=CONFIG["system"]["template_pair_source"],
    )
    parser.add_argument("--template-target-mode", choices=["min", "mean"], default=CONFIG["system"]["template_target_mode"])
    parser.add_argument("--double-bond-min", type=float, default=CONFIG["system"]["double_bond_c_c_range_ang"][0])
    parser.add_argument("--double-bond-max", type=float, default=CONFIG["system"]["double_bond_c_c_range_ang"][1])
    parser.add_argument("--existing-c-o-cutoff", type=float, default=CONFIG["system"]["existing_c_o_bond_cutoff_ang"])
    parser.add_argument("--functional-o-c-cutoff", type=float, default=CONFIG["system"]["functional_o_c_cutoff_ang"])
    parser.add_argument("--functional-c-c-shell-cutoff", type=float, default=CONFIG["system"]["functional_c_c_shell_cutoff_ang"])
    parser.add_argument("--functional-c-c-pair-cutoff", type=float, default=CONFIG["system"]["functional_c_c_pair_cutoff_ang"])
    parser.add_argument("--o-h-cutoff", type=float, default=CONFIG["system"]["o_h_bond_cutoff_ang"])
    parser.add_argument("--enol-c-o-cutoff", type=float, default=CONFIG["system"]["enol_c_o_bond_cutoff_ang"])
    parser.add_argument("--gamma", type=float, default=CONFIG["tdbb"]["gamma"])
    parser.add_argument("--f1-max", type=float, default=CONFIG["tdbb"]["f1_max"])
    parser.add_argument("--f2", type=float, default=CONFIG["tdbb"]["f2"])
    parser.add_argument("--target-scale", type=float, default=CONFIG["tdbb"]["target_scale"])
    parser.add_argument("--default-target", type=float, default=CONFIG["tdbb"]["default_target"])
    parser.add_argument("--calc-mode", default=CONFIG["pfp"]["calc_mode"])
    parser.add_argument("--interface-bonds", action=argparse.BooleanOptionalAction, default=CONFIG["interface_bonds"]["enabled"])
    parser.add_argument("--molecule-seed-symbols", default=CONFIG["interface_bonds"]["molecule_seed_symbols"])
    parser.add_argument("--molecule-symbols", default=CONFIG["interface_bonds"]["molecule_symbols"])
    parser.add_argument("--molecule-bond-symbols", default=CONFIG["interface_bonds"]["molecule_bond_symbols"])
    parser.add_argument("--surface-symbols", default=CONFIG["interface_bonds"]["surface_symbols"])
    parser.add_argument("--interface-bond-cutoff-scale", type=float, default=CONFIG["interface_bonds"]["bond_cutoff_scale"])
    parser.add_argument("--interface-min-cutoff", type=float, default=CONFIG["interface_bonds"]["min_bond_cutoff_ang"])
    parser.add_argument("--interface-max-cutoff", type=float, default=CONFIG["interface_bonds"]["max_bond_cutoff_ang"])
    parser.add_argument("--interface-min-bonds", type=int, default=CONFIG["interface_bonds"]["min_bonds_to_stabilize"])
    parser.add_argument("--interface-detection-interval", type=int, default=CONFIG["interface_bonds"]["detection_interval"])
    parser.add_argument("--interface-stable-steps", type=int, default=CONFIG["interface_bonds"]["stable_steps"])
    parser.add_argument("--interface-restraint-k", type=float, default=CONFIG["interface_bonds"]["restraint_k_ev_a2"])
    parser.add_argument("--interface-events", default=CONFIG["interface_bonds"]["event_name"])
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown command-line arguments: {unknown}")
    return args


def run_simulation(args: argparse.Namespace) -> None:
    input_path = resolve_existing_path(args.input, "Input structure", script_file=SCRIPT_FILE)
    output_root = resolve_output_path(args.output_root)
    deepmd_root = output_root / CONFIG["output"]["deepmd_dir"]
    cif_root = output_root / CONFIG["output"]["cif_dir"]
    restart_root = output_root / CONFIG["output"]["restart_dir"]
    cif_root.mkdir(parents=True, exist_ok=True)
    restart_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / args.template_progress
    interface_event_path = output_root / args.interface_events

    system = CONFIG["system"]
    relaxation = CONFIG["relaxation"]
    ctrl = CONFIG["md_control"]
    output = CONFIG["output"]

    params = TDBBParameters(
        gamma_kcal_mol_ps=args.gamma,
        f1_max_kcal_mol=args.f1_max,
        f2_inv_ang2=args.f2,
        target_scale=args.target_scale,
        default_target_ang=args.default_target,
    )
    base_calculator = build_pfp_calculator(args.calc_mode)

    if args.restart_from:
        atoms, restart_state = load_restart(
            args.restart_from,
            base_calculator,
            pbc=system["pbc"],
            fixed_z_lower=system["fixed_z_lower_bound"],
            fixed_z_upper=system["fixed_z_upper_bound"],
            script_file=SCRIPT_FILE,
        )
        start_step = int(restart_state["step"])
    else:
        atoms = read(input_path)
        atoms.calc = base_calculator
        atoms.pbc = system["pbc"]
        relax_surface(
            atoms,
            fixed_z_lower=system["fixed_z_lower_bound"],
            fixed_z_upper=system["fixed_z_upper_bound"],
            surface_depth=system["surface_relax_depth"],
            fmax=relaxation["surface_fmax"],
        )
        relax_whole_structure(
            atoms,
            fixed_z_lower=system["fixed_z_lower_bound"],
            fixed_z_upper=system["fixed_z_upper_bound"],
            fmax=relaxation["whole_fmax"],
        )
        fixed = get_aimd_fixed_indices(
            atoms,
            system["fixed_z_lower_bound"],
            system["fixed_z_upper_bound"],
        )
        atoms.set_constraint(FixAtoms(indices=fixed))
        start_step = 0

    template_data = load_template_data(args, script_file=SCRIPT_FILE) if args.pair_mode == "template" else None
    pairs = get_reactive_pairs(atoms, args)
    target_distances = get_template_target_distances(atoms, pairs, template_data, args) if template_data is not None else None
    target_keys = (
        [get_template_target_key(atoms, pair, template_data, args) for pair in pairs]
        if template_data is not None
        else ["auto"] * len(pairs)
    )

    bias = TDBBBias(atoms, pairs, params, target_distances=target_distances)
    interface_config = InterfaceBondConfig(
        enabled=args.interface_bonds,
        molecule_seed_symbols=parse_symbol_list(args.molecule_seed_symbols),
        molecule_symbols=parse_symbol_list(args.molecule_symbols),
        molecule_bond_symbols=parse_symbol_list(args.molecule_bond_symbols),
        surface_symbols=parse_symbol_list(args.surface_symbols),
        bond_cutoff_scale=args.interface_bond_cutoff_scale,
        min_bond_cutoff_ang=args.interface_min_cutoff,
        max_bond_cutoff_ang=args.interface_max_cutoff,
        min_bonds_to_stabilize=args.interface_min_bonds,
        detection_interval=max(1, args.interface_detection_interval),
        stable_steps=max(0, args.interface_stable_steps),
        restraint_k_ev_a2=args.interface_restraint_k,
        event_name=args.interface_events,
    )
    interface_stabilizer = InterfaceBondStabilizer(
        k_ev_a2=interface_config.restraint_k_ev_a2,
        hold_steps=interface_config.stable_steps,
    )
    combined_bias = CombinedBias([bias, interface_stabilizer]) if interface_config.enabled else bias
    atoms.calc = BiasedCalculator(base_calculator, combined_bias)
    writer = DeepMDWriter(atoms, deepmd_root, set_interval=output["deepmd_set_interval"])

    if not args.restart_from:
        MaxwellBoltzmannDistribution(atoms, temperature_K=ctrl["initial_temp"])

    dyn = NVTBerendsen(
        atoms,
        timestep=ctrl["timestep"] * fs,
        temperature_K=ctrl["initial_temp"],
        taut=ctrl["tau_t"],
    )

    step_counter = {"step": start_step}
    run_state = {
        "phase": "not_started",
        "target_temperature": ctrl["initial_temp"],
    }

    def update_bias_time() -> None:
        bias.set_time(step_counter["step"] * ctrl["timestep"] / 1000.0)
        if atoms.calc is not None:
            atoms.calc.reset()

    if template_data is not None:
        distance_headers = [f"d_{i}_{j}_A" for i, j in pairs]
        target_headers = [f"target_{i}_{j}_A" for i, j in pairs]
        type_headers = [f"type_{i}_{j}" for i, j in pairs]
        if not progress_path.exists() or start_step == 0:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                ",".join(["step", "time_ps", "mean_abs_delta_A", *distance_headers, *target_headers, *type_headers]) + "\n"
            )
    if interface_config.enabled and (not interface_event_path.exists() or start_step == 0):
        write_interface_event_header(interface_event_path)

    def save_frame() -> None:
        step_counter["step"] += 1
        update_bias_time()

        step = step_counter["step"]
        time_ps = step * ctrl["timestep"] / 1000.0

        if interface_config.enabled and interface_stabilizer.clear_if_expired(step):
            if atoms.calc is not None:
                atoms.calc.reset()
            print(f"Interface stabilization released at step {step}")

        if interface_config.enabled and step % interface_config.detection_interval == 0:
            interface_bonds = detect_interface_bonds(atoms, interface_config)
            activated = interface_stabilizer.update(
                atoms,
                step,
                interface_bonds,
                interface_config.min_bonds_to_stabilize,
            )
            if activated and atoms.calc is not None:
                atoms.calc.reset()
                event_energy = float(atoms.get_potential_energy())
                event_temperature = float(atoms.get_temperature())
                print(
                    f"Interface stabilization activated at step {step}: "
                    f"{len(interface_bonds)} bonds, hold until step {interface_stabilizer.active_until_step}"
                )
                append_interface_event(
                    interface_event_path,
                    step,
                    time_ps,
                    float(run_state["target_temperature"]),
                    event_temperature,
                    event_energy,
                    interface_bonds,
                )

        if step % output["save_interval"] == 0:
            writer.add_frame(atoms, step)

            set_id = step // output["cif_set_interval"]
            cif_set_dir = cif_root / f"set_{set_id:03d}"
            cif_set_dir.mkdir(parents=True, exist_ok=True)
            write(cif_set_dir / f"step_{step:08d}.cif", atoms)

            if template_data is not None:
                distances = current_pair_distances(atoms, pairs)
                targets = np.asarray(target_distances, dtype=float)
                mean_abs_delta = float(np.mean(np.abs(distances - targets)))
                row = [
                    str(step),
                    f"{time_ps:.6f}",
                    f"{mean_abs_delta:.6f}",
                    *[f"{distance:.6f}" for distance in distances],
                    *[f"{target:.6f}" for target in targets],
                    *target_keys,
                ]
                with progress_path.open("a") as file:
                    file.write(",".join(row) + "\n")

        if step % output["restart_interval"] == 0:
            save_restart(
                atoms,
                restart_root,
                step,
                run_state["target_temperature"],
                run_state["phase"],
                timestep_fs=ctrl["timestep"],
                tau_t_fs=ctrl["tau_t"],
                input_file=args.input,
                calc_mode=args.calc_mode,
            )

    dyn.attach(save_frame, interval=1)

    print("Starting shared-helper TDBB AIMD workflow")
    print(f"Input: {input_path}")
    print(f"Pair mode: {args.pair_mode}")
    print(f"Reactive pairs: {pairs}")
    if args.pair_mode == "template":
        print(f"Template directory: {args.template_dir}")
        print(f"Template bond symbols: {args.template_bond_symbols}")
        print(f"Template pair source: {args.template_pair_source}")
        print(f"Template target mode: {args.template_target_mode}")
        print(f"Template progress: {args.template_progress}")
    print(f"Target distances (A): {[round(x, 3) for x in bias.target_distances]}")
    print(f"Target bond types: {list(zip(pairs, target_keys))}")
    if interface_config.enabled:
        print(
            "Interface bond stabilization: "
            f"molecule atoms {interface_config.molecule_bond_symbols} -> "
            f"surface atoms {interface_config.surface_symbols}, "
            f"trigger >= {interface_config.min_bonds_to_stabilize} bonds, "
            f"hold {interface_config.stable_steps} steps, "
            f"k = {interface_config.restraint_k_ev_a2} eV/A^2"
        )
        print(f"Interface bond events: {interface_event_path}")
    print(f"PFP calc mode: {args.calc_mode}")
    print(f"Output root: {output_root}")

    completed_steps = start_step
    cumulative_steps = 0
    for segment in make_md_segments(
        initial_temp=ctrl["initial_temp"],
        final_temp=ctrl["final_temp"],
        ramp_interval=ctrl["ramp_interval"],
        ramp_steps=ctrl["ramp_steps"],
        stab_steps=ctrl["stab_steps"],
        prod_steps=ctrl["prod_steps"],
    ):
        segment_start = cumulative_steps
        segment_end = cumulative_steps + int(segment["steps"])
        cumulative_steps = segment_end

        if completed_steps >= segment_end:
            continue

        remaining_steps = segment_end - max(completed_steps, segment_start)
        run_state["phase"] = str(segment["phase"])
        run_state["target_temperature"] = float(segment["temperature"])
        dyn.set_temperature(temperature_K=float(segment["temperature"]))

        print(
            f"{segment['phase']} at {segment['temperature']} K: "
            f"run {remaining_steps} steps "
            f"(global {step_counter['step']} -> {segment_end})"
        )
        dyn.run(remaining_steps)
        completed_steps = segment_end


def run_jupyter(**overrides: object) -> argparse.Namespace:
    """Run from a notebook without relying on command-line arguments.

    Example:
        run_jupyter(input="ketone.cif", output_root="acid_AIMD_dataset_test")
    """
    args = parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AttributeError(f"Unknown argument: {key}")
        setattr(args, key, value)
    run_simulation(args)
    return args

def main() -> None:
    run_simulation(parse_args())


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
