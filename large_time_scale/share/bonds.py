from __future__ import annotations

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
