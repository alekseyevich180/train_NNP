from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import csv
import shutil
import sys
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read


# ============================================================
# Jupyter control switch
# ============================================================
# This is a standalone single-file version. Edit CONFIG, set RUN=True,
# then click Run in Jupyter. Or keep RUN=False and call run() manually.
RUN = False
QUIET = False
SHOW_TRACEBACK = False
SCRIPT_FILE = globals().get("__file__")


# ============================================================
# Editable CONFIG
# ============================================================
CONFIG = {
    "input": {
        # Run the script from the folder that contains generated structure subfolders.
        "root": ".",
        # Optional subdirectory under root. Keep None to scan root recursively.
        "structure_dir": None,
        "patterns": "*.cif",
        "exclude_dirs": "interface_bond_cifs",
    },
    "output": {
        "selected_dir": "interface_bond_cifs",
        "summary": "interface_bond_summary.csv",
        "folder_progress": "folder_progress.csv",
        "dry_run": False,
    },
    "performance": {
        # This machine has 6 physical cores / 12 logical threads.
        # 6 is a good default for CIF parsing without saturating memory and disk I/O.
        # 0 means use all available CPU cores.
        "workers": 6,
        # Windows/Jupyter multiprocessing is fragile when this file is run as a cell.
        # Keep True for reliable notebook runs; use command line for fastest parallel scans.
        "jupyter_force_serial": True,
    },
    "interface_bonds": {
        "molecule_seed_symbols": "C,H",
        "molecule_symbols": "C,H,O,N,S",
        "molecule_bond_symbols": "C",
        "surface_symbols": "Zn,O",
        "bond_cutoff_scale": 1.25,
        "min_bond_cutoff_ang": 0.7,
        "max_bond_cutoff_ang": 2.4,
        "min_bonds": 2,
    },
}


def config_to_namespace(config: dict) -> argparse.Namespace:
    input_config = config["input"]
    output_config = config["output"]
    performance_config = config["performance"]
    interface_config = config["interface_bonds"]
    return argparse.Namespace(
        input_root=input_config["root"],
        structure_dir=input_config["structure_dir"],
        patterns=input_config["patterns"],
        exclude_dirs=input_config["exclude_dirs"],
        output_dir=output_config["selected_dir"],
        summary=output_config["summary"],
        folder_progress=output_config["folder_progress"],
        dry_run=output_config["dry_run"],
        workers=performance_config["workers"],
        min_interface_bonds=interface_config["min_bonds"],
        molecule_seed_symbols=interface_config["molecule_seed_symbols"],
        molecule_symbols=interface_config["molecule_symbols"],
        molecule_bond_symbols=interface_config["molecule_bond_symbols"],
        surface_symbols=interface_config["surface_symbols"],
        interface_bond_cutoff_scale=interface_config["bond_cutoff_scale"],
        interface_min_cutoff=interface_config["min_bond_cutoff_ang"],
        interface_max_cutoff=interface_config["max_bond_cutoff_ang"],
    )


def run_search(config: dict | None = None) -> dict[str, object]:
    """Jupyter-friendly entry point.

    Example:
        CONFIG["input"]["root"] = "."
        CONFIG["performance"]["workers"] = 6
        result = run_search()
    """
    result = run(quiet=False, show_traceback=True, config=config)
    if result is None:
        raise RuntimeError("Search failed. Set SHOW_TRACEBACK=True for details.")
    return result


def run(
    quiet: bool = QUIET,
    show_traceback: bool = SHOW_TRACEBACK,
    config: dict | None = None,
) -> dict[str, object] | None:
    active_config = config or CONFIG
    args = config_to_namespace(active_config)

    if "ipykernel" in sys.modules and active_config["performance"].get("jupyter_force_serial", True):
        if args.workers != 1 and not quiet:
            print("Jupyter detected: using workers=1 for reliable execution.")
            print("For maximum speed, run this file from PowerShell with --workers 6.")
        args.workers = 1

    stdout_context = redirect_stdout(StringIO()) if quiet else nullcontext()
    stderr_context = redirect_stderr(StringIO()) if quiet else nullcontext()
    try:
        with stdout_context, stderr_context:
            return filter_cif_files(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        if show_traceback:
            raise
    return None


def parse_symbol_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def parse_text_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def pair_key(symbol_a: str, symbol_b: str) -> str:
    return "-".join(sorted((symbol_a, symbol_b)))


@dataclass(frozen=True)
class InterfaceBondConfig:
    molecule_seed_symbols: tuple[str, ...] = ("C", "H")
    molecule_symbols: tuple[str, ...] = ("C", "H", "O", "N", "S")
    molecule_bond_symbols: tuple[str, ...] = ("C",)
    surface_symbols: tuple[str, ...] = ("Zn", "O")
    bond_cutoff_scale: float = 1.25
    min_bond_cutoff_ang: float = 0.7
    max_bond_cutoff_ang: float = 2.4
    min_bonds: int = 2


@dataclass(frozen=True)
class InterfaceBond:
    molecule_index: int
    surface_index: int
    distance_ang: float
    symbol_pair: str


@dataclass(frozen=True)
class StructureResult:
    path: str
    selected: bool
    bond_count: int
    bonds: tuple[InterfaceBond, ...]
    error: str = ""


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
            "No molecule seed atoms were found. Adjust --molecule-seed-symbols."
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


def resolve_scan_root(input_root: Path, structure_dir: str | None) -> Path:
    if structure_dir:
        structure_path = Path(structure_dir)
        return structure_path if structure_path.is_absolute() else input_root / structure_path
    return input_root


def is_excluded(path: Path, scan_root: Path, exclude_dirs: set[str]) -> bool:
    try:
        relative = path.relative_to(scan_root)
    except ValueError:
        return False
    return any(part in exclude_dirs for part in relative.parts[:-1])


def iter_structure_files(scan_root: Path, patterns: Sequence[str], exclude_dirs: Sequence[str]) -> list[Path]:
    excluded = set(exclude_dirs)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(
            path
            for path in scan_root.rglob(pattern)
            if path.is_file() and not is_excluded(path, scan_root, excluded)
        )
    return sorted(set(files))


def bond_text(bonds: Sequence[InterfaceBond]) -> str:
    return ";".join(
        f"{bond.molecule_index}-{bond.surface_index}:{bond.symbol_pair}:{bond.distance_ang:.4f}"
        for bond in bonds
    )


def copy_selected_structure(source: Path, scan_root: Path, output_dir: Path) -> Path:
    relative = source.relative_to(scan_root)
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def analyze_structure_file(path_text: str, config: InterfaceBondConfig) -> StructureResult:
    bonds: list[InterfaceBond] = []
    try:
        atoms = read(path_text)
        bonds = detect_interface_bonds(atoms, config)
        return StructureResult(
            path=path_text,
            selected=len(bonds) >= config.min_bonds,
            bond_count=len(bonds),
            bonds=tuple(bonds),
        )
    except Exception as exc:  # Keep scanning after one bad structure.
        return StructureResult(
            path=path_text,
            selected=False,
            bond_count=len(bonds),
            bonds=tuple(bonds),
            error=f"{type(exc).__name__}: {exc}",
        )


def group_files_by_folder(paths: Sequence[Path]) -> list[tuple[Path, list[Path]]]:
    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.parent, []).append(path)
    return [(folder, sorted(files)) for folder, files in sorted(grouped.items())]


def resolve_workers(workers: int) -> int:
    if workers < 0:
        raise ValueError("--workers must be >= 0.")
    if workers == 0:
        import os

        return max(1, os.cpu_count() or 1)
    return max(1, workers)


def analyze_folder(
    files: Sequence[Path],
    config: InterfaceBondConfig,
    workers: int,
    executor: ProcessPoolExecutor | None = None,
) -> list[StructureResult]:
    if workers == 1 or len(files) <= 1:
        return [analyze_structure_file(str(path), config) for path in files]

    if executor is None:
        with ProcessPoolExecutor(max_workers=workers) as local_executor:
            return list(
                local_executor.map(
                    analyze_structure_file,
                    [str(path) for path in files],
                    [config] * len(files),
                )
            )
    return list(
        executor.map(
            analyze_structure_file,
            [str(path) for path in files],
            [config] * len(files),
        )
    )


def filter_cif_files(args: argparse.Namespace) -> dict[str, object]:
    input_root = Path(args.input_root).resolve()
    scan_root = resolve_scan_root(input_root, args.structure_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary).resolve()
    folder_progress_path = Path(args.folder_progress).resolve()

    if not scan_root.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {scan_root}")

    config = InterfaceBondConfig(
        molecule_seed_symbols=parse_symbol_list(args.molecule_seed_symbols),
        molecule_symbols=parse_symbol_list(args.molecule_symbols),
        molecule_bond_symbols=parse_symbol_list(args.molecule_bond_symbols),
        surface_symbols=parse_symbol_list(args.surface_symbols),
        bond_cutoff_scale=args.interface_bond_cutoff_scale,
        min_bond_cutoff_ang=args.interface_min_cutoff,
        max_bond_cutoff_ang=args.interface_max_cutoff,
        min_bonds=args.min_interface_bonds,
    )

    structure_files = iter_structure_files(
        scan_root,
        parse_text_list(args.patterns),
        parse_text_list(args.exclude_dirs),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    folder_progress_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    workers = resolve_workers(args.workers)
    folders = group_files_by_folder(structure_files)
    selected_count = 0
    error_count = 0
    processed_count = 0
    started_at = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        with summary_path.open("w", newline="") as summary_file, folder_progress_path.open("w", newline="") as progress_file:
            summary_writer = csv.writer(summary_file)
            progress_writer = csv.writer(progress_file)
            summary_writer.writerow(["source", "selected", "bond_count", "bonds", "copied_to", "error"])
            progress_writer.writerow(
                [
                    "folder",
                    "folder_index",
                    "folder_count",
                    "files",
                    "selected",
                    "errors",
                    "cumulative_files",
                    "elapsed_sec",
                ]
            )

            for folder_index, (folder, folder_files) in enumerate(folders, start=1):
                folder_started_at = time.perf_counter()
                results = analyze_folder(folder_files, config, workers, executor)
                folder_selected = 0
                folder_errors = 0

                for result in results:
                    copied_to = ""
                    structure_file = Path(result.path)
                    if result.selected:
                        selected_count += 1
                        folder_selected += 1
                        if not args.dry_run:
                            copied_to = str(copy_selected_structure(structure_file, scan_root, output_dir))
                    if result.error:
                        error_count += 1
                        folder_errors += 1

                    summary_writer.writerow(
                        [
                            result.path,
                            int(result.selected),
                            result.bond_count,
                            bond_text(result.bonds),
                            copied_to,
                            result.error,
                        ]
                    )

                processed_count += len(folder_files)
                elapsed = time.perf_counter() - started_at
                folder_elapsed = time.perf_counter() - folder_started_at
                progress_writer.writerow(
                    [
                        str(folder),
                        folder_index,
                        len(folders),
                        len(folder_files),
                        folder_selected,
                        folder_errors,
                        processed_count,
                        f"{elapsed:.2f}",
                    ]
                )
                summary_file.flush()
                progress_file.flush()
                print(
                    f"[{folder_index}/{len(folders)}] {folder} done: "
                    f"{len(folder_files)} files, {folder_selected} selected, "
                    f"{folder_errors} errors, {folder_elapsed:.1f}s"
                )
    finally:
        if executor is not None:
            executor.shutdown()

    print(f"Structure root: {scan_root}")
    print(f"Structure patterns: {args.patterns}")
    print(f"Excluded directories: {args.exclude_dirs}")
    print(f"Workers: {workers}")
    print(f"Scanned structure files: {len(structure_files)}")
    print(f"Selected files with >= {config.min_bonds} interface bonds: {selected_count}")
    print(f"Read/detection errors: {error_count}")
    print(f"Summary CSV: {summary_path}")
    print(f"Folder progress CSV: {folder_progress_path}")
    if not args.dry_run:
        print(f"Selected CIF output: {output_dir}")

    return {
        "scan_root": scan_root,
        "structure_files": len(structure_files),
        "selected_files": selected_count,
        "errors": error_count,
        "summary": summary_path,
        "folder_progress": folder_progress_path,
        "output_dir": output_dir if not args.dry_run else None,
        "workers": workers,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter generated structures in the current folder tree and keep structures "
            "with molecule-surface interface bonds."
        )
    )
    parser.add_argument(
        "--input-root",
        "--dataset-root",
        dest="input_root",
        default=CONFIG["input"]["root"],
        help="Root directory to scan recursively. Default is the current working directory.",
    )
    parser.add_argument(
        "--structure-dir",
        "--cif-dir",
        dest="structure_dir",
        default=CONFIG["input"]["structure_dir"],
        help="Optional structure directory relative to input root. If omitted, scan input root.",
    )
    parser.add_argument(
        "--patterns",
        default=CONFIG["input"]["patterns"],
        help="Comma-separated file patterns to scan, for example '*.cif,*.traj'.",
    )
    parser.add_argument(
        "--exclude-dirs",
        default=CONFIG["input"]["exclude_dirs"],
        help="Comma-separated directory names to skip while scanning.",
    )
    parser.add_argument("--output-dir", default=CONFIG["output"]["selected_dir"])
    parser.add_argument("--summary", default=CONFIG["output"]["summary"])
    parser.add_argument("--folder-progress", default=CONFIG["output"]["folder_progress"])
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG["performance"]["workers"],
        help="Parallel worker processes. Use 1 for serial, 0 for all CPU cores.",
    )
    parser.add_argument("--min-interface-bonds", type=int, default=CONFIG["interface_bonds"]["min_bonds"])
    parser.add_argument("--molecule-seed-symbols", default=CONFIG["interface_bonds"]["molecule_seed_symbols"])
    parser.add_argument("--molecule-symbols", default=CONFIG["interface_bonds"]["molecule_symbols"])
    parser.add_argument("--molecule-bond-symbols", default=CONFIG["interface_bonds"]["molecule_bond_symbols"])
    parser.add_argument("--surface-symbols", default=CONFIG["interface_bonds"]["surface_symbols"])
    parser.add_argument(
        "--interface-bond-cutoff-scale",
        type=float,
        default=CONFIG["interface_bonds"]["bond_cutoff_scale"],
    )
    parser.add_argument("--interface-min-cutoff", type=float, default=CONFIG["interface_bonds"]["min_bond_cutoff_ang"])
    parser.add_argument("--interface-max-cutoff", type=float, default=CONFIG["interface_bonds"]["max_bond_cutoff_ang"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=CONFIG["output"]["dry_run"],
        help="Only write the summary CSV; do not copy selected CIF files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    filter_cif_files(parse_args(argv))


if RUN:
    SEARCH_RESULT = run()
elif __name__ == "__main__" and "ipykernel" not in sys.modules and SCRIPT_FILE is not None:
    main()
