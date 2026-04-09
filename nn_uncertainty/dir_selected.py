from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


STEP_PATTERN = re.compile(r"step_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected DeepMD set directories based on frame labels from "
            "selected_structures.csv."
        )
    )
    parser.add_argument(
        "--selected-csv",
        default="outputs/selected_structures.csv",
        help="Path to selected_structures.csv. Default: outputs/selected_structures.csv",
    )
    parser.add_argument(
        "--source-deepmd",
        required=True,
        help="Path to the source deepmd_dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where matched DeepMD set folders will be copied.",
    )
    parser.add_argument(
        "--summary-name",
        default="copied_sets_summary.csv",
        help="Summary CSV filename written into the output directory.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    selected_csv = (root / args.selected_csv).resolve()
    source_deepmd = Path(args.source_deepmd).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not selected_csv.exists():
        raise SystemExit(f"selected_structures CSV not found: {selected_csv}")
    if not source_deepmd.exists():
        raise SystemExit(f"source deepmd_dataset not found: {source_deepmd}")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_sets = load_selected_set_names(selected_csv)

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

    print(f"Selected CSV: {selected_csv}")
    print(f"Source DeepMD: {source_deepmd}")
    print(f"Output directory: {output_dir}")
    print(f"Copied set folders: {copied}")
    print(f"Missing set folders: {missing}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
