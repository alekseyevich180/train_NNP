from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read


SUPPORTED_SUFFIXES = {".cif", ".xyz", ".traj", ".vasp", ".poscar", ".pdb"}


@dataclass(frozen=True)
class TargetBond:
    bond_type: str
    i: int
    j: int
    distance: float
    motif: str = ""
    segment: int | None = None
    atom_path: str = ""


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


def find_structure_file(root: Path, file_name: str) -> Path:
    requested = Path(file_name)
    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend([root / requested, root / requested.name])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and (path.name == file_name or path.stem == file_name or path.name == requested.name)
    ]
    if not matches:
        raise FileNotFoundError(f"Configured structure file was not found in {root}: {file_name}")
    return sorted(matches)[0]


def find_structure_files(root: Path, structure: dict[str, Any]) -> list[Path]:
    if "file" in structure:
        return [find_structure_file(root, str(structure["file"]).strip())]

    if "files" in structure:
        files_value = structure["files"]
        if isinstance(files_value, str):
            if files_value.lower() in {"all", "*"}:
                return iter_structure_files(root, ("*.cif", "*.xyz", "*.traj", "*.vasp", "*.poscar", "*.pdb"), set())
            patterns = parse_list(files_value)
        elif isinstance(files_value, list):
            resolved: list[Path] = []
            for item in files_value:
                text = str(item).strip()
                if any(char in text for char in "*?[]"):
                    resolved.extend(iter_structure_files(root, (text,), set()))
                else:
                    resolved.append(find_structure_file(root, text))
            return sorted(set(resolved))
        else:
            raise ValueError(f"Invalid files field in target config: {files_value}")
        return iter_structure_files(root, patterns, set())

    if "patterns" in structure:
        patterns_value = structure["patterns"]
        patterns = parse_list(patterns_value) if isinstance(patterns_value, str) else tuple(str(item) for item in patterns_value)
        return iter_structure_files(root, patterns, set())

    raise ValueError(f"Structure entry must define 'file', 'files', or 'patterns': {structure}")


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


def extract_target_bonds(atoms: Atoms, args: argparse.Namespace) -> list[TargetBond]:
    symbols = atoms.get_chemical_symbols()
    requested = set(parse_list(args.bond_types))
    region = find_enol_adsorption_region(atoms, args) if args.region_mode == "enol-adsorption" else set()
    bonds: list[TargetBond] = []

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
            bonds.append(TargetBond(bond_type=bond_type, i=i, j=j, distance=distance))

    return sorted(bonds, key=lambda item: (item.bond_type, item.i, item.j))


def infer_config_bond_type(atoms: Atoms, i: int, j: int, bond_type: str | None) -> str:
    if bond_type:
        return bond_type.strip()
    symbols = atoms.get_chemical_symbols()
    if symbols[i] == "C" and symbols[j] == "C":
        return "C=C"
    return "-".join(sorted((symbols[i], symbols[j])))


def ordered_pair_type(atoms: Atoms, i: int, j: int) -> str:
    symbols = atoms.get_chemical_symbols()
    if symbols[i] == "C" and symbols[j] == "C":
        return "C=C"
    return f"{symbols[i]}-{symbols[j]}"


def read_config_bond(row: Any) -> tuple[str | None, int, int]:
    if isinstance(row, dict):
        bond_type = row.get("bond_type", row.get("type"))
        if "atoms" in row:
            atoms = row["atoms"]
            if not isinstance(atoms, list | tuple) or len(atoms) != 2:
                raise ValueError(f"Invalid atoms field in target config: {row}")
            return (str(bond_type).strip() if bond_type else None, int(atoms[0]), int(atoms[1]))
        return (str(bond_type).strip() if bond_type else None, int(row["i"]), int(row["j"]))
    if isinstance(row, list | tuple):
        if len(row) == 2:
            return None, int(row[0]), int(row[1])
        if len(row) >= 3:
            return str(row[0]).strip(), int(row[1]), int(row[2])
    raise ValueError(f"Invalid bond entry in target config: {row}")


def read_motif_atoms(row: Any) -> tuple[str, list[int], list[str]]:
    if isinstance(row, dict):
        name = str(row.get("name", row.get("motif", ""))).strip()
        atoms = row.get("atoms")
        if not isinstance(atoms, list | tuple) or len(atoms) < 2:
            raise ValueError(f"Motif entry must contain at least two atom indices: {row}")
        bond_types = row.get("bond_types", row.get("types", []))
        if isinstance(bond_types, str):
            bond_types = parse_list(bond_types)
        return name, [int(index) for index in atoms], [str(item).strip() for item in bond_types]

    if isinstance(row, list | tuple):
        if len(row) < 2:
            raise ValueError(f"Motif entry must contain at least two atom indices: {row}")
        return "", [int(index) for index in row], []

    raise ValueError(f"Invalid motif entry in target config: {row}")


def config_bonds_for_structure(
    atoms: Atoms,
    file_path: Path,
    structure: dict[str, Any],
) -> list[TargetBond]:
    natoms = len(atoms)
    bonds: list[TargetBond] = []

    for row in structure.get("bonds", []):
        bond_type, i, j = read_config_bond(row)
        if i == j or not (0 <= i < natoms and 0 <= j < natoms):
            raise ValueError(
                f"Invalid atom indices {i}-{j} for {file_path.name}; valid range is 0-{natoms - 1}."
            )
        final_type = infer_config_bond_type(atoms, i, j, bond_type)
        distance = float(atoms.get_distance(i, j, mic=True))
        bonds.append(TargetBond(bond_type=final_type, i=i, j=j, distance=distance))

    for row in structure.get("motifs", []):
        motif_name, atom_indices, bond_types = read_motif_atoms(row)
        atom_path = "-".join(str(index) for index in atom_indices)
        for index in atom_indices:
            if not (0 <= index < natoms):
                raise ValueError(
                    f"Invalid atom index {index} in motif {atom_path} for {file_path.name}; "
                    f"valid range is 0-{natoms - 1}."
                )
        if bond_types and len(bond_types) != len(atom_indices) - 1:
            raise ValueError(
                f"Motif {atom_path} for {file_path.name} has {len(atom_indices) - 1} segments "
                f"but {len(bond_types)} bond_types."
            )

        for segment, (i, j) in enumerate(zip(atom_indices, atom_indices[1:]), start=1):
            bond_type = bond_types[segment - 1] if bond_types else ordered_pair_type(atoms, i, j)
            distance = float(atoms.get_distance(i, j, mic=True))
            bonds.append(
                TargetBond(
                    bond_type=bond_type,
                    i=i,
                    j=j,
                    distance=distance,
                    motif=motif_name,
                    segment=segment,
                    atom_path=atom_path,
                )
            )

    return sorted(bonds, key=lambda item: (item.motif, item.segment or 0, item.bond_type, item.i, item.j))


def load_config_targets(
    config_path: Path,
    interm_dir: Path,
) -> tuple[list[Path], dict[Path, list[TargetBond]]]:
    config = json.loads(config_path.read_text())
    structures = config.get("structures")
    if not isinstance(structures, list) or not structures:
        raise ValueError(f"{config_path} must contain a non-empty 'structures' list.")

    files: list[Path] = []
    bonds_by_file: dict[Path, list[TargetBond]] = {}

    for structure in structures:
        if not isinstance(structure, dict):
            raise ValueError(f"Each structure entry must be an object: {structure}")
        for file_path in find_structure_files(interm_dir, structure):
            atoms = read(file_path)
            bonds = config_bonds_for_structure(atoms, file_path, structure)
            if file_path not in bonds_by_file:
                files.append(file_path)
                bonds_by_file[file_path] = []
            bonds_by_file[file_path].extend(bonds)

    return files, bonds_by_file


def write_outputs(
    files: Sequence[Path],
    bonds_by_file: dict[Path, list[TargetBond]],
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
        writer.writerow(["file", "motif", "segment", "atom_path", "bond_type", "i", "j", "distance_A"])
        for path in output_files:
            for bond in bonds_by_file[path]:
                writer.writerow(
                    [
                        path.name,
                        bond.motif,
                        "" if bond.segment is None else bond.segment,
                        bond.atom_path,
                        bond.bond_type,
                        bond.i,
                        bond.j,
                        f"{bond.distance:.6f}",
                    ]
                )

    print(f"Wrote target structures: {structures_path}")
    print(f"Wrote target bonds: {bonds_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected transition-state target structures and C-O/C=C bonds from interm."
    )
    parser.add_argument("--interm-dir", default="large_time_scale/interm")
    parser.add_argument("--config", default="target_config.json", help="JSON config in interm-dir, or an explicit path.")
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
    config_path = resolve_path(args.config) if Path(args.config).is_absolute() else interm_dir / args.config

    if config_path.exists():
        files, bonds_by_file = load_config_targets(config_path, interm_dir)
        print(f"Loaded target config: {config_path}")
    else:
        selected = read_name_selection(select_file)
        files = iter_structure_files(interm_dir, parse_list(args.patterns), selected)
        bonds_by_file: dict[Path, list[TargetBond]] = {}

        for path in files:
            atoms = read(path)
            bonds = extract_target_bonds(atoms, args)
            bonds_by_file[path] = bonds
            print(f"{path.name}: {len(bonds)} target bonds")

    write_outputs(files, bonds_by_file, output_dir, args.structures_name, args.bonds_name, args.include_empty_structures)


if __name__ == "__main__":
    main()
