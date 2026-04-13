from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import yaml


STEP_PATTERN = re.compile(r"step_(\d+)")
DATASET_PATTERN = re.compile(r"([^/\\]+_dataset)")
AIMD_ROOT_PATTERN = re.compile(r"(.*?aimd_data)[/\\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy DeepMD set directories corresponding to selected_structures.csv. "
            "By default, infer the current dataset from nn_uncertainty/config.yaml."
        )
    )
    parser.add_argument(
        "--selected-csv",
        default="outputs/selected_structures.csv",
        help="Path to selected_structures.csv relative to nn_uncertainty. Default: outputs/selected_structures.csv",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to nn_uncertainty config.yaml. Default: config.yaml",
    )
    parser.add_argument(
        "--source-deepmd",
        help=(
            "Optional explicit deepmd_dataset path. If omitted, it will be inferred "
            "from config.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional explicit output directory. If omitted, a directory named "
            "'deepmd_selected_from_nnp' will be created next to the source deepmd_dataset."
        ),
    )
    parser.add_argument(
        "--summary-name",
        default="copied_sets_summary.csv",
        help="Summary CSV filename written into the output directory.",
    )
    parser.add_argument(
        "--copy-type-files",
        action="store_true",
        help="Also copy top-level type.raw and type_map.raw into the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied set directories.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def infer_dataset_name_from_config(config_path: Path) -> str:
    config = load_yaml_config(config_path)
    input_cfg = config.get("input", {})
    for key in ("vdw_scores", "bond_events", "bond_interface_counts"):
        value = str(input_cfg.get(key, ""))
        match = DATASET_PATTERN.search(value)
        if match:
            return match.group(1)
    raise SystemExit(
        f"Could not infer dataset name from {config_path}. "
        "Expected one of the input paths to contain '*_dataset'."
    )


def infer_aimd_root_from_config(config_path: Path) -> Path:
    config = load_yaml_config(config_path)
    input_cfg = config.get("input", {})
    for key in ("vdw_scores", "bond_events", "bond_interface_counts"):
        value = str(input_cfg.get(key, ""))
        match = AIMD_ROOT_PATTERN.search(value)
        if match:
            return (config_path.parent / match.group(1)).resolve()
    raise SystemExit(
        f"Could not infer aimd_data root from {config_path}. "
        "Expected one of the input paths to contain 'aimd_data'."
    )


def extract_step_number(frame_label: str) -> int:
    match = STEP_PATTERN.search(frame_label)
    if not match:
        raise ValueError(f"Could not parse step number from frame label: {frame_label}")
    return int(match.group(1))


def load_selected_set_names(selected_csv: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    with selected_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "frame_label" not in (reader.fieldnames or []):
            raise SystemExit(f"Missing required column 'frame_label' in {selected_csv}")

        for row in reader:
            frame_label = (row.get("frame_label") or "").strip()
            if not frame_label:
                continue
            step_number = extract_step_number(frame_label)
            set_name = f"set.{step_number}"
            if set_name in seen:
                continue
            seen.add(set_name)
            rows.append((frame_label, set_name))

    return rows


def copy_type_files(source_deepmd: Path, output_dir: Path) -> None:
    for filename in ("type.raw", "type_map.raw"):
        source_file = source_deepmd / filename
        if source_file.exists():
            shutil.copy2(source_file, output_dir / filename)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    selected_csv = (root / args.selected_csv).resolve()
    config_path = (root / args.config).resolve()

    if not selected_csv.exists():
        raise SystemExit(f"selected_structures CSV not found: {selected_csv}")
    if not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")

    if args.source_deepmd:
        source_deepmd = Path(args.source_deepmd).expanduser().resolve()
        dataset_dir = source_deepmd.parent
    else:
        dataset_name = infer_dataset_name_from_config(config_path)
        aimd_root = infer_aimd_root_from_config(config_path)
        dataset_dir = aimd_root / dataset_name
        source_deepmd = dataset_dir / "deepmd_dataset"

    if not source_deepmd.exists():
        raise SystemExit(f"source deepmd_dataset not found: {source_deepmd}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_dir / "deepmd_selected_from_nnp"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_type_files:
        copy_type_files(source_deepmd, output_dir)

    selected_sets = load_selected_set_names(selected_csv)
    summary_rows: list[dict[str, str]] = []
    copied = 0
    missing = 0
    skipped = 0

    for frame_label, set_name in selected_sets:
        source_dir = source_deepmd / set_name
        destination_dir = output_dir / set_name
        if not source_dir.exists():
            summary_rows.append(
                {
                    "frame_label": frame_label,
                    "set_name": set_name,
                    "status": "missing",
                    "source_dir": str(source_dir),
                }
            )
            missing += 1
            continue

        if destination_dir.exists():
            if not args.overwrite:
                summary_rows.append(
                    {
                        "frame_label": frame_label,
                        "set_name": set_name,
                        "status": "exists_skipped",
                        "source_dir": str(source_dir),
                    }
                )
                skipped += 1
                continue
            shutil.rmtree(destination_dir)

        shutil.copytree(source_dir, destination_dir)
        summary_rows.append(
            {
                "frame_label": frame_label,
                "set_name": set_name,
                "status": "copied",
                "source_dir": str(source_dir),
            }
        )
        copied += 1

    summary_csv = output_dir / args.summary_name
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_label", "set_name", "status", "source_dir"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Config file: {config_path}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Source DeepMD: {source_deepmd}")
    print(f"Output directory: {output_dir}")
    print(f"Copied set folders: {copied}")
    print(f"Missing set folders: {missing}")
    print(f"Skipped existing folders: {skipped}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
