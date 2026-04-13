from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


STEP_PATTERN = re.compile(r"step_(\d+)")
DATASET_PATTERN = re.compile(r"([^/\\]+_dataset)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected DeepMD set directories for one or more datasets under "
            "aimd_data based on frame labels in selected_structures.csv."
        )
    )
    parser.add_argument(
        "--selected-csv",
        default="outputs/selected_structures.csv",
        help="Path to selected_structures.csv. Default: outputs/selected_structures.csv",
    )
    parser.add_argument(
        "--aimd-root",
        required=True,
        help="Path to the aimd_data root containing multiple *_dataset directories.",
    )
    parser.add_argument(
        "--output-root-name",
        default="deepmd_selected_from_nnp",
        help=(
            "Name of the output directory created inside each dataset folder. "
            "Default: deepmd_selected_from_nnp"
        ),
    )
    parser.add_argument(
        "--summary-name",
        default="copied_sets_summary.csv",
        help="Summary CSV filename written into each dataset output directory.",
    )
    parser.add_argument(
        "--copy-type-files",
        action="store_true",
        help="Also copy top-level files such as type.raw and type_map.raw into each output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied set directories.",
    )
    return parser.parse_args()


def extract_step_number(frame_label: str) -> int:
    match = STEP_PATTERN.search(frame_label)
    if not match:
        raise ValueError(f"Could not parse step number from frame label: {frame_label}")
    return int(match.group(1))


def extract_dataset_name(row: dict[str, str]) -> str:
    for key in ("structure_path", "dataset_name", "frame_label"):
        value = (row.get(key) or "").strip()
        match = DATASET_PATTERN.search(value)
        if match:
            return match.group(1)
    raise ValueError(
        "Could not infer dataset name from row. Expected a value like "
        "'intermediates_enol_dataset' in structure_path or dataset_name."
    )


def load_selected_targets(selected_csv: Path) -> dict[str, list[tuple[str, str]]]:
    targets: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()

    with selected_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "frame_label" not in fieldnames:
            raise SystemExit(f"Missing required column 'frame_label' in {selected_csv}")

        for row in reader:
            frame_label = (row.get("frame_label") or "").strip()
            if not frame_label:
                continue
            dataset_name = extract_dataset_name(row)
            set_name = f"set.{extract_step_number(frame_label)}"
            key = (dataset_name, set_name)
            if key in seen:
                continue
            seen.add(key)
            targets.setdefault(dataset_name, []).append((frame_label, set_name))

    return targets


def copy_type_files(source_deepmd: Path, output_dir: Path) -> None:
    for filename in ("type.raw", "type_map.raw"):
        source_file = source_deepmd / filename
        if source_file.exists():
            shutil.copy2(source_file, output_dir / filename)


def process_dataset(
    dataset_dir: Path,
    selected_sets: list[tuple[str, str]],
    output_root_name: str,
    summary_name: str,
    copy_type_files_enabled: bool,
    overwrite: bool,
) -> tuple[int, int]:
    source_deepmd = dataset_dir / "deepmd_dataset"
    output_dir = dataset_dir / output_root_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if copy_type_files_enabled:
        copy_type_files(source_deepmd, output_dir)

    summary_rows: list[dict[str, str]] = []
    copied = 0
    missing = 0

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
            if not overwrite:
                summary_rows.append(
                    {
                        "frame_label": frame_label,
                        "set_name": set_name,
                        "status": "exists_skipped",
                        "source_dir": str(source_dir),
                    }
                )
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

    summary_csv = output_dir / summary_name
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_label", "set_name", "status", "source_dir"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return copied, missing


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    selected_csv = (root / args.selected_csv).resolve()
    aimd_root = Path(args.aimd_root).expanduser().resolve()

    if not selected_csv.exists():
        raise SystemExit(f"selected_structures CSV not found: {selected_csv}")
    if not aimd_root.exists():
        raise SystemExit(f"aimd_data root not found: {aimd_root}")

    selected_targets = load_selected_targets(selected_csv)

    total_copied = 0
    total_missing = 0
    processed_datasets = 0

    for dataset_name, selected_sets in sorted(selected_targets.items()):
        dataset_dir = aimd_root / dataset_name
        source_deepmd = dataset_dir / "deepmd_dataset"
        if not source_deepmd.exists():
            print(f"Skipping {dataset_name}: missing {source_deepmd}")
            continue

        copied, missing = process_dataset(
            dataset_dir=dataset_dir,
            selected_sets=selected_sets,
            output_root_name=args.output_root_name,
            summary_name=args.summary_name,
            copy_type_files_enabled=args.copy_type_files,
            overwrite=args.overwrite,
        )
        processed_datasets += 1
        total_copied += copied
        total_missing += missing
        print(
            f"{dataset_name}: copied={copied}, missing={missing}, "
            f"output={dataset_dir / args.output_root_name}"
        )

    print(f"Selected CSV: {selected_csv}")
    print(f"AIMD root: {aimd_root}")
    print(f"Processed datasets: {processed_datasets}")
    print(f"Total copied set folders: {total_copied}")
    print(f"Total missing set folders: {total_missing}")


if __name__ == "__main__":
    main()
