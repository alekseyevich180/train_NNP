from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from nn_uncertainty.share.dir_selected import copy_selected_deepmd_sets
from nn_uncertainty.share.clean_vdw_csv import clean_csv, resolve_input_csv


STEP_PATTERN = re.compile(r"step_(\d+)")
SET_PATTERN = re.compile(r"(set_\d+)")


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    frame_label: str
    structure_path: str
    features: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank stable enol2 frames using vdW and optional interface features "
            "without building positive/negative labels."
        )
    )
    parser.add_argument(
        "--vdw-input",
        default="../../aimd_data/intermediates_enol_oxidacid_2H-H2O_dataset/vdw",
        help="Path to a vdW CSV file or vdW directory.",
    )
    parser.add_argument(
        "--interface-csv",
        default="",
        help="Optional bond_interface_counts.csv path.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs_oxidacid_2H-H2O_dataset",
        help="Output directory relative to nn_uncertainty.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=300,
        help="Maximum number of selected frames.",
    )
    parser.add_argument(
        "--min-step-gap",
        type=int,
        default=500,
        help="Minimum step difference between selected frames in the same set.",
    )
    parser.add_argument(
        "--copy-frames",
        action="store_true",
        help="Copy selected CIF frames into the output directory.",
    )
    parser.add_argument(
        "--copy-deepmd",
        action="store_true",
        help="Copy matching DeepMD set directories.",
    )
    return parser.parse_args()


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def parse_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def prepare_clean_vdw_csv(vdw_input_path: Path) -> Path:
    resolved_input = resolve_input_csv(vdw_input_path)
    output_name = "vdw_scores_clean.csv"
    if resolved_input.name == output_name:
        return resolved_input
    return clean_csv(resolved_input, output_name)


def load_vdw_records(vdw_csv_path: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    with vdw_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"vdW CSV is missing a header row: {vdw_csv_path}")
        metadata_columns = {
            "frame_index",
            "frame_label",
            "structure_path",
            "dataset_name",
            "previous_frame_label",
        }
        feature_columns = [name for name in reader.fieldnames if name not in metadata_columns]
        for row in reader:
            records.append(
                FrameRecord(
                    frame_index=parse_int(row.get("frame_index"), len(records)),
                    frame_label=row.get("frame_label") or "",
                    structure_path=row.get("structure_path") or "",
                    features={name: parse_float(row.get(name), 0.0) for name in feature_columns},
                )
            )
    return records


def load_interface_rows(csv_path: Path) -> dict[str, dict[str, float]]:
    if not csv_path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_label = row.get("frame_label") or ""
            if not frame_label:
                continue
            organic_surface = parse_float(row.get("organic-surface"), 0.0)
            water_surface = parse_float(row.get("water-surface"), 0.0)
            organic_water = parse_float(row.get("organic-water"), 0.0)
            rows[frame_label] = {
                "interface_organic_surface": organic_surface,
                "interface_water_surface": water_surface,
                "interface_organic_water": organic_water,
                "interface_total": organic_surface + water_surface + organic_water,
            }
    return rows


def percentile_rank_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: item[1])
    size = max(len(ordered) - 1, 1)
    ranks: dict[str, float] = {}
    for idx, (key, _) in enumerate(ordered):
        ranks[key] = idx / size
    return ranks


def parse_set_and_step(frame_label: str) -> tuple[str, int]:
    set_match = SET_PATTERN.search(frame_label)
    step_match = STEP_PATTERN.search(frame_label)
    set_name = set_match.group(1) if set_match else frame_label
    step_number = int(step_match.group(1)) if step_match else -1
    return set_name, step_number


def score_records(records: list[FrameRecord], interface_rows: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    vdw_total = {record.frame_label: record.features.get("vdw_total_score", 0.0) for record in records}
    surface_intermediate = {
        record.frame_label: record.features.get("surface_intermediate_score", 0.0)
        for record in records
    }
    dissociated_intermediate = {
        record.frame_label: record.features.get("water_dissociated_intermediate_score", 0.0)
        for record in records
    }
    interface_total = {
        record.frame_label: interface_rows.get(record.frame_label, {}).get("interface_total", 0.0)
        for record in records
    }

    vdw_rank = percentile_rank_map(vdw_total)
    surface_rank = percentile_rank_map(surface_intermediate)
    dissociated_rank = percentile_rank_map(dissociated_intermediate)
    interface_rank = percentile_rank_map(interface_total)

    scored_rows: list[dict[str, object]] = []
    for record in records:
        label = record.frame_label
        interface = interface_rows.get(label, {})
        score = (
            0.50 * vdw_rank.get(label, 0.0)
            + 0.20 * surface_rank.get(label, 0.0)
            + 0.15 * dissociated_rank.get(label, 0.0)
            + 0.15 * interface_rank.get(label, 0.0)
        )
        scored_rows.append(
            {
                "frame_index": record.frame_index,
                "frame_label": label,
                "structure_path": record.structure_path,
                "selection_score": score,
                "vdw_total_score": record.features.get("vdw_total_score", 0.0),
                "surface_intermediate_score": record.features.get("surface_intermediate_score", 0.0),
                "water_dissociated_intermediate_score": record.features.get("water_dissociated_intermediate_score", 0.0),
                "interface_total": interface.get("interface_total", 0.0),
                "interface_organic_surface": interface.get("interface_organic_surface", 0.0),
                "interface_water_surface": interface.get("interface_water_surface", 0.0),
                "interface_organic_water": interface.get("interface_organic_water", 0.0),
            }
        )
    scored_rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    return scored_rows


def select_rows(scored_rows: list[dict[str, object]], top_k: int, min_step_gap: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    last_steps: dict[str, list[int]] = {}
    for row in scored_rows:
        set_name, step_number = parse_set_and_step(str(row["frame_label"]))
        existing_steps = last_steps.setdefault(set_name, [])
        if step_number >= 0 and any(abs(step_number - existing) < min_step_gap for existing in existing_steps):
            continue
        selected.append(row)
        if step_number >= 0:
            existing_steps.append(step_number)
        if top_k > 0 and len(selected) >= top_k:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_selected_frames(selected_rows: list[dict[str, object]], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in selected_rows:
        source = Path(str(row["structure_path"]))
        if not source.exists():
            continue
        shutil.copy2(source, output_dir / source.name)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    vdw_input_path = (root / args.vdw_input).resolve()
    interface_csv_path = (root / args.interface_csv).resolve() if args.interface_csv else Path()
    output_root = (root / args.output_root).resolve()
    selected_frames_dir = output_root / "selected_frames"
    summary_path = output_root / "summary.json"
    scores_csv = output_root / "selector_scores.csv"
    selected_csv = output_root / "selected_structures.csv"

    cleaned_vdw_csv = prepare_clean_vdw_csv(vdw_input_path)
    records = load_vdw_records(cleaned_vdw_csv)
    interface_rows = load_interface_rows(interface_csv_path) if args.interface_csv else {}
    scored_rows = score_records(records, interface_rows)
    selected_rows = select_rows(scored_rows, args.top_k, args.min_step_gap)

    write_csv(
        scores_csv,
        scored_rows,
        [
            "frame_index",
            "frame_label",
            "structure_path",
            "selection_score",
            "vdw_total_score",
            "surface_intermediate_score",
            "water_dissociated_intermediate_score",
            "interface_total",
            "interface_organic_surface",
            "interface_water_surface",
            "interface_organic_water",
        ],
    )
    write_csv(
        selected_csv,
        selected_rows,
        [
            "frame_index",
            "frame_label",
            "structure_path",
            "selection_score",
            "vdw_total_score",
            "surface_intermediate_score",
            "water_dissociated_intermediate_score",
            "interface_total",
            "interface_organic_surface",
            "interface_water_surface",
            "interface_organic_water",
        ],
    )

    copied_frames = copy_selected_frames(selected_rows, selected_frames_dir) if args.copy_frames else 0

    deepmd_export: dict[str, object] = {}
    if args.copy_deepmd:
        deepmd_export = copy_selected_deepmd_sets(
            selected_csv=selected_csv,
            config_path=root / "config.yaml",
            source_deepmd=(vdw_input_path.parent.parent / "deepmd_dataset").resolve(),
            output_dir=(vdw_input_path.parent.parent / "deepmd_selected_from_enol2_ranked").resolve(),
            copy_type_files_flag=True,
            overwrite=False,
        )

    summary = {
        "module": "nn_uncertainty",
        "mode": "stable_enol2_ranked_selection",
        "vdw_input": str(vdw_input_path),
        "vdw_clean_input": str(cleaned_vdw_csv),
        "interface_csv": str(interface_csv_path) if args.interface_csv else "",
        "num_records": len(records),
        "selected_count": len(selected_rows),
        "top_k": args.top_k,
        "min_step_gap": args.min_step_gap,
        "copied_frames": copied_frames,
        "deepmd_export": deepmd_export,
        "score_formula": {
            "vdw_total_score_rank": 0.50,
            "surface_intermediate_score_rank": 0.20,
            "water_dissociated_intermediate_score_rank": 0.15,
            "interface_total_rank": 0.15,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"vdW input: {cleaned_vdw_csv}")
    print(f"Scores CSV: {scores_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Selected frames: {len(selected_rows)}")
    print(f"Copied CIF frames: {copied_frames}")
    if deepmd_export:
        print(f"DeepMD output: {deepmd_export['output_dir']}")


if __name__ == "__main__":
    main()
