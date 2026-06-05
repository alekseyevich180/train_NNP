from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read


SUPPORTED_SUFFIXES = {".cif", ".xyz", ".traj", ".vasp", ".poscar", ".pdb"}


def parse_list(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def read_name_selection(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()

    selected: set[str] = set()
    for line in path.read_text().splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        path_item = Path(item)
        selected.update({item, path_item.name, path_item.stem})
    return selected


def iter_structure_files(root: Path, patterns: Sequence[str], selected: set[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(
            path
            for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )

    unique = sorted(set(files))
    if selected:
        unique = [
            path
            for path in unique
            if path.name in selected or path.stem in selected or str(path) in selected
        ]
    if not unique:
        raise FileNotFoundError(f"No matching template structures were found in {root}.")
    return unique


def pair_key(symbol_a: str, symbol_b: str) -> str:
    if symbol_a == "C" and symbol_b == "C":
        return "C-C"
    return "-".join(sorted((symbol_a, symbol_b)))


def covalent_cutoff(atoms: Atoms, i: int, j: int, args: argparse.Namespace) -> float:
    numbers = atoms.get_atomic_numbers()
    ri = float(covalent_radii[numbers[i]])
    rj = float(covalent_radii[numbers[j]])
    if not np.isfinite(ri) or not np.isfinite(rj) or ri <= 0.0 or rj <= 0.0:
        return args.max_bond_cutoff
    cutoff = args.bond_cutoff_scale * (ri + rj)
    return min(max(cutoff, args.min_bond_cutoff), args.max_bond_cutoff)


def detect_molecule_indices(atoms: Atoms, args: argparse.Namespace) -> set[int]:
    symbols = atoms.get_chemical_symbols()
    allowed = set(parse_list(args.molecule_symbols))
    seeds = {
        index
        for index, symbol in enumerate(symbols)
        if symbol in parse_list(args.molecule_seed_symbols) and symbol in allowed
    }
    if not seeds:
        return set()

    molecule = set(seeds)
    changed = True
    while changed:
        changed = False
        for i, symbol_i in enumerate(symbols):
            if i in molecule or symbol_i not in allowed:
                continue
            for j in tuple(molecule):
                if float(atoms.get_distance(i, j, mic=True)) <= covalent_cutoff(atoms, i, j, args):
                    molecule.add(i)
                    changed = True
                    break

    return molecule


def bonded_indices(atoms: Atoms, center: int, symbol: str, cutoff: float) -> list[int]:
    symbols = atoms.get_chemical_symbols()
    return [
        index
        for index, atom_symbol in enumerate(symbols)
        if index != center
        and atom_symbol == symbol
        and float(atoms.get_distance(center, index, mic=True)) <= cutoff
    ]


def find_molecule_double_bonds(atoms: Atoms, molecule_indices: set[int], args: argparse.Namespace) -> list[tuple[int, int]]:
    symbols = atoms.get_chemical_symbols()
    carbon_indices = sorted(index for index in molecule_indices if symbols[index] == "C")
    double_bonds: list[tuple[int, int]] = []
    for pos, i in enumerate(carbon_indices):
        for j in carbon_indices[pos + 1 :]:
            distance = float(atoms.get_distance(i, j, mic=True))
            if args.c_c_double_min <= distance <= args.c_c_double_max:
                double_bonds.append((i, j))
    return double_bonds


def find_enol_adsorption_region(atoms: Atoms, args: argparse.Namespace) -> set[int]:
    symbols = atoms.get_chemical_symbols()
    molecule_indices = detect_molecule_indices(atoms, args)
    if not molecule_indices:
        return set()

    double_bonds = find_molecule_double_bonds(atoms, molecule_indices, args)
    double_bond_carbons = {index for pair in double_bonds for index in pair}
    molecule_oxygens = sorted(index for index in molecule_indices if symbols[index] == "O")
    surface_symbols = set(parse_list(args.surface_symbols))
    surface_indices = [
        index
        for index, symbol in enumerate(symbols)
        if index not in molecule_indices and symbol in surface_symbols
    ]

    region: set[int] = set()
    region_double_bonds: set[tuple[int, int]] = set()
    for oxygen in molecule_oxygens:
        bonded_h = bonded_indices(atoms, oxygen, "H", args.o_h_cutoff)
        bonded_double_c = [
            carbon
            for carbon in double_bond_carbons
            if float(atoms.get_distance(oxygen, carbon, mic=True)) <= args.enol_c_o_cutoff
        ]
        if not bonded_h and not bonded_double_c:
            continue

        nearby_surface = [
            index
            for index in surface_indices
            if float(atoms.get_distance(oxygen, index, mic=True)) <= args.surface_adsorption_cutoff
            or any(float(atoms.get_distance(carbon, index, mic=True)) <= args.surface_adsorption_cutoff for carbon in bonded_double_c)
        ]
        if args.require_surface_adsorption and not nearby_surface:
            continue

        region.add(oxygen)
        region.update(bonded_h)
        region.update(bonded_double_c)
        region.update(nearby_surface)
        for pair in double_bonds:
            if pair[0] in bonded_double_c or pair[1] in bonded_double_c:
                region.update(pair)
                region_double_bonds.add(tuple(sorted(pair)))

    # Store exact region C=C pairs on args for the current structure only.
    args._region_double_bonds = region_double_bonds
    return region


def target_bond_type(symbol_a: str, symbol_b: str, distance: float, args: argparse.Namespace) -> str | None:
    key = pair_key(symbol_a, symbol_b)
    if key == "C-O" and distance <= args.c_o_cutoff:
        return "C-O"
    if key == "C-C" and args.c_c_double_min <= distance <= args.c_c_double_max:
        return "C=C"
    return None


def is_region_target_bond(
    atoms: Atoms,
    i: int,
    j: int,
    bond_type: str,
    region: set[int],
    args: argparse.Namespace,
) -> bool:
    if args.region_mode == "all":
        return True
    if not region:
        return False

    symbols = atoms.get_chemical_symbols()
    pair = tuple(sorted((i, j)))
    if bond_type == "C=C":
        return pair in getattr(args, "_region_double_bonds", set())
    if bond_type == "C-O":
        return i in region and j in region and "C" in {symbols[i], symbols[j]}
    return False


def extract_target_bonds(atoms: Atoms, args: argparse.Namespace) -> list[tuple[str, int, int, float]]:
    symbols = atoms.get_chemical_symbols()
    requested = set(parse_list(args.bond_types))
    region = find_enol_adsorption_region(atoms, args) if args.region_mode == "enol-adsorption" else set()
    bonds: list[tuple[str, int, int, float]] = []

    for i, symbol_i in enumerate(symbols):
        for j in range(i + 1, len(symbols)):
            symbol_j = symbols[j]
            if "C" not in {symbol_i, symbol_j}:
                continue

            distance = float(atoms.get_distance(i, j, mic=True))
            bond_type = target_bond_type(symbol_i, symbol_j, distance, args)
            if bond_type is None or bond_type not in requested:
                continue
            if not is_region_target_bond(atoms, i, j, bond_type, region, args):
                continue
            bonds.append((bond_type, i, j, distance))

    return sorted(bonds, key=lambda item: (item[0], item[1], item[2]))


def write_outputs(
    files: Sequence[Path],
    bonds_by_file: dict[Path, list[tuple[str, int, int, float]]],
    output_dir: Path,
    structures_name: str,
    bonds_name: str,
    include_empty_structures: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        path
        for path in files
        if include_empty_structures or bonds_by_file[path]
    ]
    structures_path = output_dir / structures_name
    structures_path.write_text("\n".join(path.name for path in output_files) + "\n")

    bonds_path = output_dir / bonds_name
    with bonds_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "bond_type", "i", "j", "distance_A"])
        for path in output_files:
            for bond_type, i, j, distance in bonds_by_file[path]:
                writer.writerow([path.name, bond_type, i, j, f"{distance:.6f}"])

    print(f"Wrote target structures: {structures_path}")
    print(f"Wrote target bonds: {bonds_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected transition-state target structures and C-O/C=C bonds from interm."
    )
    parser.add_argument("--interm-dir", default="large_time_scale/interm")
    parser.add_argument("--patterns", default="*.cif,*.xyz,*.traj,*.vasp,*.poscar,*.pdb")
    parser.add_argument("--select-file", default=None, help="Optional text file listing template file names or stems.")
    parser.add_argument("--bond-types", default="C-O,C=C")
    parser.add_argument("--region-mode", choices=["enol-adsorption", "all"], default="enol-adsorption")
    parser.add_argument("--c-o-cutoff", type=float, default=2.2)
    parser.add_argument("--c-c-double-min", type=float, default=1.15)
    parser.add_argument("--c-c-double-max", type=float, default=1.45)
    parser.add_argument("--o-h-cutoff", type=float, default=1.2)
    parser.add_argument("--enol-c-o-cutoff", type=float, default=1.65)
    parser.add_argument("--surface-adsorption-cutoff", type=float, default=2.6)
    parser.add_argument("--require-surface-adsorption", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--molecule-seed-symbols", default="C,H")
    parser.add_argument("--molecule-symbols", default="C,H,O,N,S")
    parser.add_argument("--surface-symbols", default="Zn,O")
    parser.add_argument("--bond-cutoff-scale", type=float, default=1.25)
    parser.add_argument("--min-bond-cutoff", type=float, default=0.7)
    parser.add_argument("--max-bond-cutoff", type=float, default=2.4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--structures-name", default="target_structures.txt")
    parser.add_argument("--bonds-name", default="target_bonds.csv")
    parser.add_argument("--include-empty-structures", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interm_dir = resolve_path(args.interm_dir)
    output_dir = resolve_path(args.output_dir) if args.output_dir else interm_dir
    select_file = resolve_path(args.select_file) if args.select_file else None

    selected = read_name_selection(select_file)
    files = iter_structure_files(interm_dir, parse_list(args.patterns), selected)
    bonds_by_file: dict[Path, list[tuple[str, int, int, float]]] = {}

    for path in files:
        atoms = read(path)
        bonds = extract_target_bonds(atoms, args)
        bonds_by_file[path] = bonds
        print(f"{path.name}: {len(bonds)} target bonds")

    write_outputs(files, bonds_by_file, output_dir, args.structures_name, args.bonds_name, args.include_empty_structures)


if __name__ == "__main__":
    main()
