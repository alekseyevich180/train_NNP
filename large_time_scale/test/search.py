from __future__ import annotations

import argparse
import csv
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read


CONFIG = {
    "input": {
        "root": ".",
        "structure_dir": None,
        "patterns": "*.cif",
        "exclude_dirs": "interface_bond_cifs",
    },
    "output": {
        "selected_dir": "interface_bond_cifs",
        "summary": "interface_bond_summary.csv",
        "progress_markdown": "interface_bond_progress.md",
        "dry_run": False,
    },
    "performance": {
        "workers": 1,
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


@dataclass(frozen=True)
class InterfaceBondConfig:
    molecule_seed_symbols: tuple[str, ...]
    molecule_symbols: tuple[str, ...]
    molecule_bond_symbols: tuple[str, ...]
    surface_symbols: tuple[str, ...]
    bond_cutoff_scale: float
    min_bond_cutoff_ang: float
    max_bond_cutoff_ang: float
    min_bonds: int


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


def parse_list(text: str | Sequence[str] | None) -> tuple[str, ...]:
    if text is None:
        return ()
    if isinstance(text, str):
        return tuple(item.strip() for item in text.split(",") if item.strip())
    return tuple(str(item).strip() for item in text if str(item).strip())


def pair_key(symbol_a: str, symbol_b: str) -> str:
    return "-".join(sorted((symbol_a, symbol_b)))


def build_interface_config(config: dict) -> InterfaceBondConfig:
    interface_config = config["interface_bonds"]
    return InterfaceBondConfig(
        molecule_seed_symbols=parse_list(interface_config["molecule_seed_symbols"]),
        molecule_symbols=parse_list(interface_config["molecule_symbols"]),
        molecule_bond_symbols=parse_list(interface_config["molecule_bond_symbols"]),
        surface_symbols=parse_list(interface_config["surface_symbols"]),
        bond_cutoff_scale=float(interface_config["bond_cutoff_scale"]),
        min_bond_cutoff_ang=float(interface_config["min_bond_cutoff_ang"]),
        max_bond_cutoff_ang=float(interface_config["max_bond_cutoff_ang"]),
        min_bonds=int(interface_config["min_bonds"]),
    )


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
    molecule = {
        index
        for index, symbol in enumerate(symbols)
        if symbol in config.molecule_seed_symbols and symbol in allowed
    }
    if not molecule:
        raise ValueError("No molecule seed atoms were found.")

    changed = True
    while changed:
        changed = False
        for i, symbol_i in enumerate(symbols):
            if i in molecule or symbol_i not in allowed:
                continue
            for j in tuple(molecule):
                distance = float(atoms.get_distance(i, j, mic=True))
                if distance <= covalent_cutoff(atoms, i, j, config):
                    molecule.add(i)
                    changed = True
                    break

    return molecule


def detect_interface_bonds(atoms: Atoms, config: InterfaceBondConfig) -> tuple[InterfaceBond, ...]:
    molecule_indices = detect_molecule_indices(atoms, config)
    symbols = atoms.get_chemical_symbols()
    molecule_bond_symbols = set(config.molecule_bond_symbols)
    surface_indices = [
        index
        for index, symbol in enumerate(symbols)
        if index not in molecule_indices and symbol in config.surface_symbols
    ]

    bonds: list[InterfaceBond] = []
    for molecule_index in sorted(molecule_indices):
        if symbols[molecule_index] not in molecule_bond_symbols:
            continue
        for surface_index in surface_indices:
            distance = float(atoms.get_distance(molecule_index, surface_index, mic=True))
            if distance <= covalent_cutoff(atoms, molecule_index, surface_index, config):
                bonds.append(
                    InterfaceBond(
                        molecule_index=molecule_index,
                        surface_index=surface_index,
                        distance_ang=distance,
                        symbol_pair=pair_key(symbols[molecule_index], symbols[surface_index]),
                    )
                )

    return tuple(sorted(bonds, key=lambda bond: (bond.molecule_index, bond.surface_index)))


def resolve_scan_root(input_root: Path, structure_dir: str | None) -> Path:
    if not structure_dir:
        return input_root
    structure_path = Path(structure_dir)
    return structure_path if structure_path.is_absolute() else input_root / structure_path


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


def analyze_structure_file(path: Path, config: InterfaceBondConfig) -> StructureResult:
    bonds: tuple[InterfaceBond, ...] = ()
    try:
        atoms = read(path)
        bonds = detect_interface_bonds(atoms, config)
        return StructureResult(
            path=str(path),
            selected=len(bonds) >= config.min_bonds,
            bond_count=len(bonds),
            bonds=bonds,
        )
    except Exception as exc:
        return StructureResult(
            path=str(path),
            selected=False,
            bond_count=len(bonds),
            bonds=bonds,
            error=f"{type(exc).__name__}: {exc}",
        )


def bond_text(bonds: Sequence[InterfaceBond]) -> str:
    return ";".join(
        f"{bond.molecule_index}-{bond.surface_index}:{bond.symbol_pair}:{bond.distance_ang:.4f}"
        for bond in bonds
    )


def unique_destination(output_dir: Path, filename: str) -> Path:
    destination = output_dir / filename
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    index = 2
    while True:
        candidate = output_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_selected_structure(source: Path, output_dir: Path) -> Path:
    destination = unique_destination(output_dir, source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def resolve_workers(workers: int) -> int:
    if workers < 0:
        raise ValueError("--workers must be >= 0.")
    if workers == 0:
        return max(1, os.cpu_count() or 1)
    return max(1, workers)


def iter_parallel_results(
    structure_files: Sequence[Path],
    config: InterfaceBondConfig,
    workers: int,
) -> Sequence[tuple[Path, StructureResult]]:
    max_pending = max(workers * 2, workers)
    path_iter = iter(structure_files)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {}
        for path in path_iter:
            pending[executor.submit(analyze_structure_file, path, config)] = path
            if len(pending) >= max_pending:
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                path = pending.pop(future)
                yield path, future.result()

                try:
                    next_path = next(path_iter)
                except StopIteration:
                    continue
                pending[executor.submit(analyze_structure_file, next_path, config)] = next_path


def markdown_cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def display_path(path: Path, scan_root: Path) -> str:
    try:
        return str(path.relative_to(scan_root))
    except ValueError:
        return str(path)


def write_progress_markdown(
    progress_path: Path,
    scan_root: Path,
    total_count: int,
    processed_count: int,
    selected_count: int,
    error_count: int,
    current_path: Path | None,
    results: Sequence[StructureResult],
) -> None:
    current_file = display_path(current_path, scan_root) if current_path is not None else ""
    lines = [
        "# Interface Bond Search Progress",
        "",
        f"- Processed: {processed_count}/{total_count}",
        f"- Selected: {selected_count}",
        f"- Errors: {error_count}",
        f"- Current file: {current_file}",
        "",
        "| file | selected | surface bond count | bonds | error |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(display_path(Path(result.path), scan_root)),
                    str(int(result.selected)),
                    str(result.bond_count),
                    markdown_cell(bond_text(result.bonds)),
                    markdown_cell(result.error),
                ]
            )
            + " |"
        )

    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def filter_structure_files(config: dict, show_progress: bool = True) -> dict[str, object]:
    input_config = config["input"]
    output_config = config["output"]
    performance_config = config["performance"]

    input_root = Path(input_config["root"]).resolve()
    scan_root = resolve_scan_root(input_root, input_config["structure_dir"]).resolve()
    output_dir = Path(output_config["selected_dir"]).resolve()
    summary_path = Path(output_config["summary"]).resolve()
    progress_path = Path(output_config["progress_markdown"]).resolve()
    dry_run = bool(output_config["dry_run"])
    interface_config = build_interface_config(config)
    workers = resolve_workers(int(performance_config["workers"]))

    if not scan_root.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {scan_root}")

    structure_files = iter_structure_files(
        scan_root,
        parse_list(input_config["patterns"]),
        parse_list(input_config["exclude_dirs"]),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    selected_count = 0
    error_count = 0
    processed_results: list[StructureResult] = []
    write_progress_markdown(
        progress_path,
        scan_root,
        len(structure_files),
        0,
        selected_count,
        error_count,
        None,
        processed_results,
    )
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["source", "selected", "bond_count", "bonds", "copied_to", "error"])

        if workers == 1 or len(structure_files) <= 1:
            result_iter = (
                (path, analyze_structure_file(path, interface_config))
                for path in structure_files
            )
        else:
            result_iter = iter_parallel_results(structure_files, interface_config, workers)

        for processed_count, (path, result) in enumerate(result_iter, start=1):
            copied_to = ""
            if result.selected:
                selected_count += 1
                if not dry_run:
                    copied_to = str(copy_selected_structure(path, output_dir))
            if result.error:
                error_count += 1
            processed_results.append(result)

            writer.writerow(
                [
                    result.path,
                    int(result.selected),
                    result.bond_count,
                    bond_text(result.bonds),
                    copied_to,
                    result.error,
                ]
            )
            summary_file.flush()
            write_progress_markdown(
                progress_path,
                scan_root,
                len(structure_files),
                processed_count,
                selected_count,
                error_count,
                path,
                processed_results,
            )
            if show_progress:
                print(
                    f"[{processed_count}/{len(structure_files)}] "
                    f"{display_path(path, scan_root)} | bonds={result.bond_count} "
                    f"| selected={int(result.selected)}",
                    end="\r",
                    flush=True,
                )

    if show_progress:
        print()

    print(f"Structure root: {scan_root}")
    print(f"Structure patterns: {input_config['patterns']}")
    print(f"Excluded directories: {input_config['exclude_dirs']}")
    print(f"Workers: {workers}")
    print(f"Scanned structure files: {len(structure_files)}")
    print(f"Selected files with >= {interface_config.min_bonds} interface bonds: {selected_count}")
    print(f"Read/detection errors: {error_count}")
    print(f"Summary CSV: {summary_path}")
    print(f"Progress Markdown: {progress_path}")
    if not dry_run:
        print(f"Selected structure output: {output_dir}")

    return {
        "scan_root": scan_root,
        "structure_files": len(structure_files),
        "selected_files": selected_count,
        "errors": error_count,
        "summary": summary_path,
        "progress_markdown": progress_path,
        "output_dir": output_dir if not dry_run else None,
        "workers": workers,
    }


def config_from_args(args: argparse.Namespace) -> dict:
    return {
        "input": {
            "root": args.input_root,
            "structure_dir": args.structure_dir,
            "patterns": args.patterns,
            "exclude_dirs": args.exclude_dirs,
        },
        "output": {
            "selected_dir": args.output_dir,
            "summary": args.summary,
            "progress_markdown": args.progress_markdown,
            "dry_run": args.dry_run,
        },
        "performance": {
            "workers": args.workers,
        },
        "interface_bonds": {
            "molecule_seed_symbols": args.molecule_seed_symbols,
            "molecule_symbols": args.molecule_symbols,
            "molecule_bond_symbols": args.molecule_bond_symbols,
            "surface_symbols": args.surface_symbols,
            "bond_cutoff_scale": args.interface_bond_cutoff_scale,
            "min_bond_cutoff_ang": args.interface_min_cutoff,
            "max_bond_cutoff_ang": args.interface_max_cutoff,
            "min_bonds": args.min_interface_bonds,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter structure files with molecule-surface interface bonds."
    )
    parser.add_argument("--input-root", "--dataset-root", dest="input_root", default=CONFIG["input"]["root"])
    parser.add_argument("--structure-dir", "--cif-dir", dest="structure_dir", default=CONFIG["input"]["structure_dir"])
    parser.add_argument("--patterns", default=CONFIG["input"]["patterns"])
    parser.add_argument("--exclude-dirs", default=CONFIG["input"]["exclude_dirs"])
    parser.add_argument("--output-dir", default=CONFIG["output"]["selected_dir"])
    parser.add_argument("--summary", default=CONFIG["output"]["summary"])
    parser.add_argument("--progress-markdown", default=CONFIG["output"]["progress_markdown"])
    parser.add_argument("--dry-run", action="store_true", default=CONFIG["output"]["dry_run"])
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG["performance"]["workers"],
        help="Parallel Python worker processes. Use 1 for serial, 0 for all CPU cores.",
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    filter_structure_files(config_from_args(parse_args(argv)), show_progress=True)


if __name__ == "__main__":
    main()
