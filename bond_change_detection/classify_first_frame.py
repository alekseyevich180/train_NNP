from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ase.io import iread, read
from ase.neighborlist import neighbor_list

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency issue is environment-specific
    raise SystemExit("PyYAML is required to run classify_first_frame.") from exc


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def normalize_pair(symbol_a: str, symbol_b: str) -> tuple[str, str]:
    return tuple(sorted((symbol_a, symbol_b)))


def parse_cutoffs(raw_cutoffs: dict[str, float]) -> dict[tuple[str, str], float]:
    parsed: dict[tuple[str, str], float] = {}
    for pair_label, cutoff in raw_cutoffs.items():
        left, right = pair_label.split("-", maxsplit=1)
        parsed[normalize_pair(left, right)] = float(cutoff)
    return parsed


def parse_selection(selection: dict[str, Any]) -> dict[str, int | None]:
    start_frame = max(0, int(selection.get("start_frame", 0)))
    end_frame_raw = selection.get("end_frame")
    end_frame = None if end_frame_raw in (None, "") else max(start_frame, int(end_frame_raw))
    frame_stride = max(1, int(selection.get("frame_stride", 1)))
    max_frames_raw = selection.get("max_frames")
    max_frames = None if max_frames_raw in (None, "") else max(1, int(max_frames_raw))
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
    }


def select_entries(entries: list[Any], selection: dict[str, Any]) -> list[Any]:
    parsed = parse_selection(selection)
    start_frame = int(parsed["start_frame"] or 0)
    end_frame = parsed["end_frame"]
    frame_stride = int(parsed["frame_stride"] or 1)
    max_frames = parsed["max_frames"]

    selected = entries[start_frame:end_frame:frame_stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def discover_frame_paths(input_path: Path, input_format: str, selection: dict[str, Any]) -> list[Path]:
    if input_format == "cif_dir":
        if not input_path.exists():
            return []
        return select_entries(sorted(input_path.rglob("*.cif")), selection)
    raise ValueError(f"Frame paths are only supported for cif_dir input, got: {input_format}")


def load_first_frame(input_path: Path, input_format: str, selection: dict[str, Any]) -> tuple[str, Any]:
    if input_format == "cif_dir":
        frame_paths = discover_frame_paths(input_path, input_format, selection)
        if not frame_paths:
            raise SystemExit(f"No input frames found in {input_path}")
        frame_path = frame_paths[0]
        return frame_path.stem, read(frame_path)

    if input_format == "trajectory_file":
        parsed = parse_selection(selection)
        start_frame = int(parsed["start_frame"] or 0)
        end_frame = parsed["end_frame"]
        frame_stride = int(parsed["frame_stride"] or 1)
        max_frames = parsed["max_frames"]
        selected_count = 0

        for idx, atoms in enumerate(iread(input_path, index=":")):
            if idx < start_frame:
                continue
            if end_frame is not None and idx >= end_frame:
                break
            if (idx - start_frame) % frame_stride != 0:
                continue
            if max_frames is not None and selected_count >= max_frames:
                break
            return f"frame_{idx:08d}", atoms

        raise SystemExit(f"No selected frames found in {input_path}")

    raise ValueError(f"Unsupported input format: {input_format}")


def build_bond_map(atoms: Any, cutoffs: dict[tuple[str, str], float]) -> dict[tuple[int, int], float]:
    neighbor_cutoffs = {(left, right): cutoff for (left, right), cutoff in cutoffs.items()}
    indices_i, indices_j, _distances = neighbor_list("ijd", atoms, neighbor_cutoffs)

    bond_map: dict[tuple[int, int], float] = {}
    symbols = atoms.get_chemical_symbols()
    for atom_i, atom_j in zip(indices_i, indices_j):
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if atom_i >= atom_j:
            continue
        pair_key = normalize_pair(symbols[atom_i], symbols[atom_j])
        if pair_key not in cutoffs:
            continue
        bond_map[(atom_i, atom_j)] = 1.0
    return bond_map


def build_adjacency(atom_count: int, bond_map: dict[tuple[int, int], float]) -> list[set[int]]:
    adjacency = [set() for _ in range(atom_count)]
    for atom_i, atom_j in bond_map:
        adjacency[atom_i].add(atom_j)
        adjacency[atom_j].add(atom_i)
    return adjacency


def classify_initial_atoms(atoms: Any, bond_map: dict[tuple[int, int], float]) -> tuple[dict[int, str], dict[str, list[int]]]:
    symbols = atoms.get_chemical_symbols()
    adjacency = build_adjacency(len(symbols), bond_map)
    atom_groups: dict[int, str] = {index: "unassigned" for index in range(len(symbols))}
    grouped_indices: dict[str, list[int]] = {
        "surface": [],
        "organic": [],
        "water": [],
        "unassigned": [],
    }

    surface_indices: set[int] = set()
    organic_indices: set[int] = set()
    water_indices: set[int] = set()

    for atom_index, symbol in enumerate(symbols):
        if symbol != "Zn":
            continue
        surface_indices.add(atom_index)
        for neighbor in adjacency[atom_index]:
            if symbols[neighbor] == "O":
                surface_indices.add(neighbor)

    for atom_index, symbol in enumerate(symbols):
        if symbol != "O" or atom_index in surface_indices:
            continue
        hydrogen_neighbors = [neighbor for neighbor in adjacency[atom_index] if symbols[neighbor] == "H"]
        carbon_neighbors = [neighbor for neighbor in adjacency[atom_index] if symbols[neighbor] == "C"]
        zinc_neighbors = [neighbor for neighbor in adjacency[atom_index] if symbols[neighbor] == "Zn"]
        if len(hydrogen_neighbors) == 2 and not carbon_neighbors and not zinc_neighbors:
            water_indices.add(atom_index)
            water_indices.update(hydrogen_neighbors)

    carbon_seed_indices = {index for index, symbol in enumerate(symbols) if symbol == "C"}
    pending = list(carbon_seed_indices)
    organic_indices.update(carbon_seed_indices)
    while pending:
        atom_index = pending.pop()
        for neighbor in adjacency[atom_index]:
            if neighbor in surface_indices or neighbor in water_indices or neighbor in organic_indices:
                continue
            if symbols[neighbor] in {"C", "H", "O"}:
                organic_indices.add(neighbor)
                pending.append(neighbor)

    for atom_index in sorted(surface_indices):
        atom_groups[atom_index] = "surface"
    for atom_index in sorted(water_indices - surface_indices):
        atom_groups[atom_index] = "water"
    for atom_index in sorted(organic_indices - surface_indices - water_indices):
        atom_groups[atom_index] = "organic"

    for atom_index, label in atom_groups.items():
        grouped_indices[label].append(atom_index)

    for label in grouped_indices:
        grouped_indices[label].sort()
    return atom_groups, grouped_indices


def export_group_json(grouped_indices: dict[str, list[int]], output_path: Path) -> None:
    output_path.write_text(json.dumps(grouped_indices, indent=2), encoding="utf-8")


def export_group_csv(atom_groups: dict[int, str], atoms: Any, output_path: Path) -> None:
    fieldnames = ["atom_index", "symbol", "group"]
    symbols = atoms.get_chemical_symbols()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for atom_index in range(len(symbols)):
            writer.writerow(
                {
                    "atom_index": atom_index,
                    "symbol": symbols[atom_index],
                    "group": atom_groups[atom_index],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify the first selected frame into surface, water, and organic groups.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("bond_change_detection/outputs/atom_groups_first_frame.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("bond_change_detection/outputs/atom_groups_first_frame.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    root = config_path.parent
    config = load_config(config_path)

    input_path = (root / config["input"]["trajectory"]).resolve()
    input_format = config["input"]["format"]
    cutoffs = parse_cutoffs(config["chemistry"]["cutoffs"])
    selection = config.get("selection", {})

    frame_label, atoms = load_first_frame(input_path, input_format, selection)
    bond_map = build_bond_map(atoms, cutoffs)
    atom_groups, grouped_indices = classify_initial_atoms(atoms, bond_map)

    json_out = args.json_out.resolve()
    csv_out = args.csv_out.resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    export_group_json(grouped_indices, json_out)
    export_group_csv(atom_groups, atoms, csv_out)

    print(f"First frame classified: {frame_label}")
    print(f"surface: {len(grouped_indices['surface'])}")
    print(f"organic: {len(grouped_indices['organic'])}")
    print(f"water: {len(grouped_indices['water'])}")
    print(f"unassigned: {len(grouped_indices['unassigned'])}")
    print(f"JSON: {json_out}")
    print(f"CSV: {csv_out}")


if __name__ == "__main__":
    main()
