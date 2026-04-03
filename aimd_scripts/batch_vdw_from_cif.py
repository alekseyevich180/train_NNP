from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list


CONFIG = {
    "batch": {
        # Search root containing folders such as intermediates_enol_dataset
        "search_root": ".",
        "dataset_glob": "*_dataset",
        "cif_dir_name": "cif_frames",
        "vdw_dir_name": "vdw",
    },
    "selection": {
        "max_frames_per_dataset": 4000,
    },
    "vdw": {
        "bond_cutoff_scale": 1.1,
        "contact_factor": 1.35,
        "close_contact_ratio": 1.05,
        "fallback_min_distance": 10.0,
        "close_contact_weight": 0.15,
        "contact_density_weight": 0.30,
        "pair_weights": {
            "surface_water": 1.0,
            "surface_intermediate": 1.3,
            "water_intermediate": 0.8,
        },
    },
}


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
    ("surface", "water_intact"),
    ("surface", "water_dissociated"),
    ("surface", "intermediate"),
    ("water_intact", "intermediate"),
    ("water_dissociated", "intermediate"),
)


@dataclass(frozen=True)
class FrameScore:
    dataset_name: str
    frame_label: str
    structure_path: str
    features: dict[str, float]


@dataclass(frozen=True)
class ReferenceGroups:
    atom_groups: dict[int, str]
    water_oxygen_indices: set[int]
    water_members_by_oxygen: dict[int, set[int]]


def discover_dataset_dirs(search_root: Path, dataset_glob: str, cif_dir_name: str) -> list[Path]:
    dataset_dirs: list[Path] = []
    for candidate in sorted(search_root.glob(dataset_glob)):
        cif_dir = candidate / cif_dir_name
        if candidate.is_dir() and cif_dir.exists():
            dataset_dirs.append(candidate)
    return dataset_dirs


def frame_label_from_path(cif_root: Path, cif_path: Path) -> str:
    relative_path = cif_path.relative_to(cif_root)
    return str(relative_path.with_suffix("")).replace("\\", "__").replace("/", "__")


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


def classify_initial_atoms(atoms: Any, adjacency: list[set[int]]) -> ReferenceGroups:
    symbols = atoms.get_chemical_symbols()
    atom_groups: dict[int, str] = {index: "unassigned" for index in range(len(symbols))}

    surface_indices: set[int] = set()
    water_indices: set[int] = set()
    intermediate_indices: set[int] = set()
    water_oxygen_indices: set[int] = set()
    water_members_by_oxygen: dict[int, set[int]] = {}

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
            water_oxygen_indices.add(atom_index)
            water_members_by_oxygen[atom_index] = {atom_index, *hydrogen_neighbors}

    carbon_seed_indices = {index for index, symbol in enumerate(symbols) if symbol == "C"}
    intermediate_indices.update(carbon_seed_indices)
    pending = list(carbon_seed_indices)
    while pending:
        atom_index = pending.pop()
        for neighbor in adjacency[atom_index]:
            if neighbor in surface_indices or neighbor in water_indices or neighbor in intermediate_indices:
                continue
            if symbols[neighbor] in {"C", "H", "O"}:
                intermediate_indices.add(neighbor)
                pending.append(neighbor)

    for atom_index in surface_indices:
        atom_groups[atom_index] = "surface"
    for atom_index in water_indices - surface_indices:
        atom_groups[atom_index] = "water_intact"
    for atom_index in intermediate_indices - surface_indices - water_indices:
        atom_groups[atom_index] = "intermediate"
    return ReferenceGroups(
        atom_groups=atom_groups,
        water_oxygen_indices=water_oxygen_indices,
        water_members_by_oxygen=water_members_by_oxygen,
    )


def build_frame_groups(
    atoms: Any,
    adjacency: list[set[int]],
    reference: ReferenceGroups,
) -> dict[int, str]:
    symbols = atoms.get_chemical_symbols()
    atom_groups = dict(reference.atom_groups)

    for oxygen_index in reference.water_oxygen_indices:
        members = reference.water_members_by_oxygen.get(oxygen_index, {oxygen_index})
        hydrogen_neighbors = [neighbor for neighbor in adjacency[oxygen_index] if symbols[neighbor] == "H"]
        carbon_neighbors = [neighbor for neighbor in adjacency[oxygen_index] if symbols[neighbor] == "C"]
        zinc_neighbors = [neighbor for neighbor in adjacency[oxygen_index] if symbols[neighbor] == "Zn"]
        is_intact = len(hydrogen_neighbors) == 2 and not carbon_neighbors and not zinc_neighbors

        if is_intact:
            atom_groups[oxygen_index] = "water_intact"
            for member in members:
                if symbols[member] == "H":
                    atom_groups[member] = "water_intact"
        else:
            atom_groups[oxygen_index] = "water_dissociated"
            for member in members:
                if symbols[member] == "H":
                    atom_groups[member] = "water_dissociated"

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


def compute_vdw_features(atoms: Any, reference: ReferenceGroups) -> dict[str, float]:
    vdw_cfg = CONFIG["vdw"]
    symbols = atoms.get_chemical_symbols()
    adjacency = build_adjacency(atoms, float(vdw_cfg["bond_cutoff_scale"]))
    atom_groups = build_frame_groups(atoms, adjacency, reference)

    contact_factor = float(vdw_cfg["contact_factor"])
    close_contact_ratio = float(vdw_cfg["close_contact_ratio"])
    fallback_min_distance = float(vdw_cfg["fallback_min_distance"])
    per_atom_cutoffs = [VDW_RADII.get(symbol, 1.70) * contact_factor for symbol in symbols]
    indices_i, indices_j, distances = neighbor_list("ijd", atoms, per_atom_cutoffs)

    pair_metrics: dict[str, dict[str, float]] = {}
    for left, right in TRACKED_PAIRS:
        pair_metrics[f"{left}_{right}"] = {
            "min_distance": fallback_min_distance,
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

    features: dict[str, float] = {}
    pair_weights = vdw_cfg["pair_weights"]
    total_score = 0.0
    for left, right in TRACKED_PAIRS:
        pair_key = f"{left}_{right}"
        metric = pair_metrics[pair_key]
        score = (
            float(metric["lj_sum"])
            + float(metric["close_contacts"]) * float(vdw_cfg["close_contact_weight"])
            + float(metric["contact_density"]) * float(vdw_cfg["contact_density_weight"])
        )
        score *= float(pair_weights.get(pair_key, 1.0))
        total_score += score

        features[f"{pair_key}_min_distance"] = float(metric["min_distance"])
        features[f"{pair_key}_lj_sum"] = float(metric["lj_sum"])
        features[f"{pair_key}_close_contacts"] = float(metric["close_contacts"])
        features[f"{pair_key}_contact_density"] = float(metric["contact_density"])
        features[f"{pair_key}_score"] = float(score)

    features["vdw_total_score"] = float(total_score)
    features["surface_atom_count"] = float(sum(1 for value in atom_groups.values() if value == "surface"))
    features["water_intact_atom_count"] = float(sum(1 for value in atom_groups.values() if value == "water_intact"))
    features["water_dissociated_atom_count"] = float(sum(1 for value in atom_groups.values() if value == "water_dissociated"))
    features["intermediate_atom_count"] = float(sum(1 for value in atom_groups.values() if value == "intermediate"))
    return features


def collect_frame_scores(dataset_dir: Path) -> list[FrameScore]:
    cif_root = dataset_dir / CONFIG["batch"]["cif_dir_name"]
    cif_paths = sorted(cif_root.rglob("*.cif"))
    max_frames = CONFIG["selection"]["max_frames_per_dataset"]
    if max_frames is not None:
        cif_paths = cif_paths[: int(max_frames)]

    if not cif_paths:
        return []

    first_atoms = read(cif_paths[0])
    first_adjacency = build_adjacency(first_atoms, float(CONFIG["vdw"]["bond_cutoff_scale"]))
    reference_groups = classify_initial_atoms(first_atoms, first_adjacency)

    scores: list[FrameScore] = []
    for cif_path in cif_paths:
        atoms = read(cif_path)
        frame_label = frame_label_from_path(cif_root, cif_path)
        scores.append(
            FrameScore(
                dataset_name=dataset_dir.name,
                frame_label=frame_label,
                structure_path=str(cif_path.resolve()),
                features=compute_vdw_features(atoms, reference_groups),
            )
        )
    return scores


def export_dataset_scores(dataset_dir: Path, scores: list[FrameScore]) -> None:
    output_dir = dataset_dir / CONFIG["batch"]["vdw_dir_name"]
    output_dir.mkdir(exist_ok=True)

    summary_csv = output_dir / "vdw_summary.csv"
    metadata_json = output_dir / "metadata.json"

    fieldnames = [
        "dataset_name",
        "frame_label",
        "structure_path",
        "surface_water_intact_score",
        "surface_water_dissociated_score",
        "surface_intermediate_score",
        "water_intact_intermediate_score",
        "water_dissociated_intermediate_score",
        "vdw_total_score",
        "surface_water_intact_min_distance",
        "surface_water_dissociated_min_distance",
        "surface_intermediate_min_distance",
        "water_intact_intermediate_min_distance",
        "water_dissociated_intermediate_min_distance",
        "surface_water_intact_lj_sum",
        "surface_water_dissociated_lj_sum",
        "surface_intermediate_lj_sum",
        "water_intact_intermediate_lj_sum",
        "water_dissociated_intermediate_lj_sum",
        "surface_water_intact_close_contacts",
        "surface_water_dissociated_close_contacts",
        "surface_intermediate_close_contacts",
        "water_intact_intermediate_close_contacts",
        "water_dissociated_intermediate_close_contacts",
        "surface_atom_count",
        "water_intact_atom_count",
        "water_dissociated_atom_count",
        "intermediate_atom_count",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "dataset_name": score.dataset_name,
                    "frame_label": score.frame_label,
                    "structure_path": score.structure_path,
                    "surface_water_intact_score": f"{score.features['surface_water_intact_score']:.6f}",
                    "surface_water_dissociated_score": f"{score.features['surface_water_dissociated_score']:.6f}",
                    "surface_intermediate_score": f"{score.features['surface_intermediate_score']:.6f}",
                    "water_intact_intermediate_score": f"{score.features['water_intact_intermediate_score']:.6f}",
                    "water_dissociated_intermediate_score": f"{score.features['water_dissociated_intermediate_score']:.6f}",
                    "vdw_total_score": f"{score.features['vdw_total_score']:.6f}",
                    "surface_water_intact_min_distance": f"{score.features['surface_water_intact_min_distance']:.6f}",
                    "surface_water_dissociated_min_distance": f"{score.features['surface_water_dissociated_min_distance']:.6f}",
                    "surface_intermediate_min_distance": f"{score.features['surface_intermediate_min_distance']:.6f}",
                    "water_intact_intermediate_min_distance": f"{score.features['water_intact_intermediate_min_distance']:.6f}",
                    "water_dissociated_intermediate_min_distance": f"{score.features['water_dissociated_intermediate_min_distance']:.6f}",
                    "surface_water_intact_lj_sum": f"{score.features['surface_water_intact_lj_sum']:.6f}",
                    "surface_water_dissociated_lj_sum": f"{score.features['surface_water_dissociated_lj_sum']:.6f}",
                    "surface_intermediate_lj_sum": f"{score.features['surface_intermediate_lj_sum']:.6f}",
                    "water_intact_intermediate_lj_sum": f"{score.features['water_intact_intermediate_lj_sum']:.6f}",
                    "water_dissociated_intermediate_lj_sum": f"{score.features['water_dissociated_intermediate_lj_sum']:.6f}",
                    "surface_water_intact_close_contacts": f"{score.features['surface_water_intact_close_contacts']:.0f}",
                    "surface_water_dissociated_close_contacts": f"{score.features['surface_water_dissociated_close_contacts']:.0f}",
                    "surface_intermediate_close_contacts": f"{score.features['surface_intermediate_close_contacts']:.0f}",
                    "water_intact_intermediate_close_contacts": f"{score.features['water_intact_intermediate_close_contacts']:.0f}",
                    "water_dissociated_intermediate_close_contacts": f"{score.features['water_dissociated_intermediate_close_contacts']:.0f}",
                    "surface_atom_count": f"{score.features['surface_atom_count']:.0f}",
                    "water_intact_atom_count": f"{score.features['water_intact_atom_count']:.0f}",
                    "water_dissociated_atom_count": f"{score.features['water_dissociated_atom_count']:.0f}",
                    "intermediate_atom_count": f"{score.features['intermediate_atom_count']:.0f}",
                }
            )

    stats = {
        "frames_processed": len(scores),
        "score_ranges": {},
    }
    for column in (
        "surface_water_intact_score",
        "surface_water_dissociated_score",
        "surface_intermediate_score",
        "water_intact_intermediate_score",
        "water_dissociated_intermediate_score",
        "vdw_total_score",
    ):
        values = [score.features[column] for score in scores]
        if values:
            stats["score_ranges"][column] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
        else:
            stats["score_ranges"][column] = {"min": 0.0, "max": 0.0, "mean": 0.0}

    metadata = {
        "dataset_dir": str(dataset_dir.resolve()),
        "cif_dir": str((dataset_dir / CONFIG["batch"]["cif_dir_name"]).resolve()),
        "output_dir": str(output_dir.resolve()),
        "tracked_pairs": [
            "surface-water_intact",
            "surface-water_dissociated",
            "surface-intermediate",
            "water_intact-intermediate",
            "water_dissociated-intermediate",
        ],
        "vdw_parameters": CONFIG["vdw"],
        **stats,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run() -> None:
    if "__file__" in globals():
        script_root = Path(__file__).resolve().parent
    else:
        script_root = Path.cwd()
    search_root = (script_root / CONFIG["batch"]["search_root"]).resolve()
    dataset_dirs = discover_dataset_dirs(
        search_root,
        str(CONFIG["batch"]["dataset_glob"]),
        str(CONFIG["batch"]["cif_dir_name"]),
    )
    if not dataset_dirs:
        print(f"No dataset directories found under {search_root}")
        return

    batch_summary: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        print(f"Processing {dataset_dir.name}")
        scores = collect_frame_scores(dataset_dir)
        export_dataset_scores(dataset_dir, scores)
        batch_summary.append(
            {
                "dataset_name": dataset_dir.name,
                "dataset_dir": str(dataset_dir.resolve()),
                "frames_processed": len(scores),
                "vdw_dir": str((dataset_dir / CONFIG["batch"]["vdw_dir_name"]).resolve()),
            }
        )

    summary_path = search_root / "batch_vdw_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"Processed {len(dataset_dirs)} datasets. Summary written to {summary_path}")


if __name__ == "__main__":
    run()
