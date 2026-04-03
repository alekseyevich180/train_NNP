from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency issue is environment-specific
    raise SystemExit("PyYAML is required to run vdw_energy_predictor.") from exc

try:
    from ase.io import iread, read
    from ase.neighborlist import natural_cutoffs, neighbor_list
except ImportError as exc:  # pragma: no cover - dependency issue is environment-specific
    raise SystemExit("ASE is required to run vdw_energy_predictor.") from exc


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "O": 1.52,
    "Zn": 1.39,
}

VDW_EPSILON = {
    "H": 0.030,
    "C": 0.105,
    "O": 0.060,
    "Zn": 0.124,
}

TRACKED_PAIRS = (
    ("organic", "surface"),
    ("water", "surface"),
    ("organic", "water"),
)


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    frame_label: str
    structure_path: str
    features: dict[str, float]


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


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


def frame_label_from_path(input_path: Path, frame_file: Path) -> str:
    relative_path = frame_file.relative_to(input_path)
    return str(relative_path.with_suffix("")).replace("\\", "__").replace("/", "__")


def discover_frame_paths(input_path: Path, input_format: str, selection: dict[str, Any]) -> list[Path]:
    if input_format == "cif_dir":
        if not input_path.exists():
            return []
        return select_entries(sorted(input_path.rglob("*.cif")), selection)
    raise ValueError(f"Frame paths are only supported for cif_dir input, got: {input_format}")


def iter_frames(input_path: Path, input_format: str, selection: dict[str, Any]):
    if input_format == "cif_dir":
        for frame_file in discover_frame_paths(input_path, input_format, selection):
            yield frame_label_from_path(input_path, frame_file), frame_file, read(frame_file)
        return

    if input_format == "trajectory_file":
        if not input_path.exists():
            return
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
            yield f"frame_{idx:08d}", input_path, atoms
            selected_count += 1
        return

    raise ValueError(f"Unsupported input format: {input_format}")


def build_adjacency(atoms: Any, cutoff_scale: float) -> list[set[int]]:
    per_atom_cutoffs = natural_cutoffs(atoms, mult=cutoff_scale)
    indices_i, indices_j = neighbor_list("ij", atoms, per_atom_cutoffs)
    adjacency = [set() for _ in range(len(atoms))]
    for atom_i, atom_j in zip(indices_i, indices_j):
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if atom_i == atom_j:
            continue
        adjacency[atom_i].add(atom_j)
        adjacency[atom_j].add(atom_i)
    return adjacency


def classify_atoms(atoms: Any, adjacency: list[set[int]]) -> dict[int, str]:
    symbols = atoms.get_chemical_symbols()
    atom_groups: dict[int, str] = {index: "unassigned" for index in range(len(symbols))}

    surface_indices: set[int] = set()
    water_indices: set[int] = set()
    organic_indices: set[int] = set()

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
    organic_indices.update(carbon_seed_indices)
    pending = list(carbon_seed_indices)
    while pending:
        atom_index = pending.pop()
        for neighbor in adjacency[atom_index]:
            if neighbor in surface_indices or neighbor in water_indices or neighbor in organic_indices:
                continue
            if symbols[neighbor] in {"C", "H", "O"}:
                organic_indices.add(neighbor)
                pending.append(neighbor)

    for atom_index in surface_indices:
        atom_groups[atom_index] = "surface"
    for atom_index in water_indices - surface_indices:
        atom_groups[atom_index] = "water"
    for atom_index in organic_indices - surface_indices - water_indices:
        atom_groups[atom_index] = "organic"
    return atom_groups


def ordered_pair(group_i: str, group_j: str) -> tuple[str, str] | None:
    for left, right in TRACKED_PAIRS:
        if {group_i, group_j} == {left, right}:
            return (left, right)
    return None


def lj_attraction(symbol_i: str, symbol_j: str, distance: float) -> float:
    radius_i = VDW_RADII.get(symbol_i, 1.70)
    radius_j = VDW_RADII.get(symbol_j, 1.70)
    epsilon_i = VDW_EPSILON.get(symbol_i, 0.080)
    epsilon_j = VDW_EPSILON.get(symbol_j, 0.080)
    r_min = radius_i + radius_j
    sigma = r_min / (2.0 ** (1.0 / 6.0))
    epsilon = math.sqrt(epsilon_i * epsilon_j)
    ratio = sigma / max(distance, 1.0e-6)
    energy = 4.0 * epsilon * ((ratio ** 12) - (ratio ** 6))
    return max(-energy, 0.0)


def compute_frame_features(atoms: Any, config: dict[str, Any]) -> dict[str, float]:
    predictor_cfg = config["predictor"]
    adjacency = build_adjacency(atoms, float(predictor_cfg.get("bond_cutoff_scale", 1.1)))
    atom_groups = classify_atoms(atoms, adjacency)
    symbols = atoms.get_chemical_symbols()

    contact_factor = float(predictor_cfg.get("contact_factor", 1.35))
    close_contact_ratio = float(predictor_cfg.get("close_contact_ratio", 1.05))
    per_atom_cutoffs = [VDW_RADII.get(symbol, 1.70) * contact_factor for symbol in symbols]
    indices_i, indices_j, distances = neighbor_list("ijd", atoms, per_atom_cutoffs)

    features: dict[str, float] = {}
    pair_metrics: dict[str, dict[str, float]] = {}
    for left, right in TRACKED_PAIRS:
        pair_key = f"{left}_{right}"
        pair_metrics[pair_key] = {
            "min_distance": float(predictor_cfg.get("fallback_min_distance", 10.0)),
            "lj_sum": 0.0,
            "close_contacts": 0.0,
            "contact_density": 0.0,
        }

    for atom_i, atom_j, distance in zip(indices_i, indices_j, distances):
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if atom_i >= atom_j:
            continue
        if atom_j in adjacency[atom_i]:
            continue

        group_i = atom_groups.get(atom_i, "unassigned")
        group_j = atom_groups.get(atom_j, "unassigned")
        tracked = ordered_pair(group_i, group_j)
        if tracked is None:
            continue

        pair_key = f"{tracked[0]}_{tracked[1]}"
        metric = pair_metrics[pair_key]
        distance = float(distance)
        metric["min_distance"] = min(metric["min_distance"], distance)

        symbol_i = symbols[atom_i]
        symbol_j = symbols[atom_j]
        vdw_distance = VDW_RADII.get(symbol_i, 1.70) + VDW_RADII.get(symbol_j, 1.70)
        metric["lj_sum"] += lj_attraction(symbol_i, symbol_j, distance)
        if distance <= vdw_distance * close_contact_ratio:
            metric["close_contacts"] += 1.0
        contact_extent = max(vdw_distance * contact_factor, 1.0e-6)
        metric["contact_density"] += max(0.0, (contact_extent - distance) / contact_extent)

    weights = predictor_cfg.get("pair_weights", {})
    total_score = 0.0
    total_contacts = 0.0
    for left, right in TRACKED_PAIRS:
        pair_key = f"{left}_{right}"
        metric = pair_metrics[pair_key]
        min_distance = float(metric["min_distance"])
        if min_distance >= float(predictor_cfg.get("fallback_min_distance", 10.0)):
            min_distance = float(predictor_cfg.get("fallback_min_distance", 10.0))
        score = (
            float(metric["lj_sum"])
            + float(metric["close_contacts"]) * float(predictor_cfg.get("close_contact_weight", 0.15))
            + float(metric["contact_density"]) * float(predictor_cfg.get("contact_density_weight", 0.30))
        )
        score *= float(weights.get(pair_key, 1.0))
        total_score += score
        total_contacts += float(metric["close_contacts"])

        features[f"{pair_key}_min_distance"] = min_distance
        features[f"{pair_key}_lj_sum"] = float(metric["lj_sum"])
        features[f"{pair_key}_close_contacts"] = float(metric["close_contacts"])
        features[f"{pair_key}_contact_density"] = float(metric["contact_density"])
        features[f"{pair_key}_score"] = score

    features["vdw_total_score"] = total_score
    features["vdw_total_close_contacts"] = total_contacts
    features["surface_atom_count"] = float(sum(1 for value in atom_groups.values() if value == "surface"))
    features["organic_atom_count"] = float(sum(1 for value in atom_groups.values() if value == "organic"))
    features["water_atom_count"] = float(sum(1 for value in atom_groups.values() if value == "water"))
    return features


def export_scores(path: Path, results: list[FrameResult]) -> None:
    fieldnames = ["frame_index", "frame_label", "structure_path", *sorted(results[0].features)] if results else [
        "frame_index",
        "frame_label",
        "structure_path",
        "vdw_total_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "frame_index": result.frame_index,
                "frame_label": result.frame_label,
                "structure_path": result.structure_path,
            }
            for key, value in result.features.items():
                row[key] = f"{float(value):.6f}"
            writer.writerow(row)


def export_selected(path: Path, selected: list[FrameResult], score_column: str) -> None:
    fieldnames = ["frame_index", "frame_label", "structure_path", score_column]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in selected:
            writer.writerow(
                {
                    "frame_index": result.frame_index,
                    "frame_label": result.frame_label,
                    "structure_path": result.structure_path,
                    score_column: f"{float(result.features[score_column]):.6f}",
                }
            )


def copy_selected_frames(selected: list[FrameResult], selected_dir: Path) -> int:
    copied = 0
    for result in selected:
        source = Path(result.structure_path)
        if not source.exists():
            continue
        shutil.copy2(source, selected_dir / source.name)
        copied += 1
    return copied


def build_summary(config: dict[str, Any], input_path: Path, results: list[FrameResult], copied_count: int) -> dict[str, Any]:
    score_column = str(config["selection"]["score_column"])
    values = [float(result.features[score_column]) for result in results]
    if values:
        score_stats = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
    else:
        score_stats = {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "module": "vdw_energy_predictor",
        "status": "completed",
        "input_path": str(input_path),
        "frames_processed": len(results),
        "score_column": score_column,
        "score_stats": score_stats,
        "copied_selected_frames": copied_count,
        "tracked_pairs": [f"{left}-{right}" for left, right in TRACKED_PAIRS],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    output_root = root / config["output"]["root"]
    output_root.mkdir(exist_ok=True)
    selected_dir = output_root / config["output"]["selected_dir"]
    selected_dir.mkdir(exist_ok=True)

    input_path = (root / config["input"]["trajectory"]).resolve()
    input_format = str(config["input"]["format"])
    selection_cfg = config.get("input", {}).get("selection", {})
    score_column = str(config["selection"]["score_column"])

    results: list[FrameResult] = []
    for frame_index, (frame_label, frame_path, atoms) in enumerate(iter_frames(input_path, input_format, selection_cfg)):
        features = compute_frame_features(atoms, config)
        structure_path = str(frame_path.resolve()) if Path(frame_path).exists() else str(frame_path)
        results.append(
            FrameResult(
                frame_index=frame_index,
                frame_label=frame_label,
                structure_path=structure_path,
                features=features,
            )
        )

    scores_csv = output_root / config["output"]["scores_csv"]
    selected_csv = output_root / config["output"]["selected_csv"]
    summary_json = output_root / config["output"]["summary_json"]

    export_scores(scores_csv, results)
    ranked = sorted(results, key=lambda item: float(item.features[score_column]), reverse=True)
    top_k = max(0, int(config["selection"].get("top_k", 0)))
    selected = ranked[:top_k] if top_k > 0 else ranked
    export_selected(selected_csv, selected, score_column)
    copied_count = copy_selected_frames(selected, selected_dir)

    summary = build_summary(config, input_path, results, copied_count)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "vdw_energy_predictor completed: "
        f"{len(results)} frames processed, {len(selected)} selected."
    )


if __name__ == "__main__":
    main()
