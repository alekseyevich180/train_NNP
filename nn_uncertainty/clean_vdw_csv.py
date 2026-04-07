from __future__ import annotations

import argparse
import csv
from pathlib import Path


CHECKPOINT_MARKERS = (".ipynb_checkpoints", "-checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean a vdW CSV by removing checkpoint-derived rows and write a new CSV "
            "into the same directory."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to a vdW CSV file or a directory containing vdW CSV files.",
    )
    parser.add_argument(
        "--output-name",
        default="vdw_scores_clean.csv",
        help="Name of the cleaned CSV written into the same directory as the input CSV.",
    )
    return parser.parse_args()


def resolve_input_csv(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")

    candidates = (
        input_path / "vdw_summary.csv",
        input_path / "vdw_scores.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Could not find vdW CSV in the given directory. Expected one of: "
        "vdw_summary.csv, vdw_scores.csv"
    )


def row_contains_checkpoint(row: dict[str, str]) -> bool:
    frame_label = row.get("frame_label", "")
    structure_path = row.get("structure_path", "")
    values = (frame_label, structure_path)
    return any(marker in value for value in values for marker in CHECKPOINT_MARKERS)


def clean_csv(input_csv: Path, output_name: str) -> Path:
    output_csv = input_csv.with_name(output_name)
    kept_rows: list[dict[str, str]] = []
    dropped_rows = 0

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit(f"CSV is missing a header row: {input_csv}")
        if "frame_label" not in fieldnames:
            raise SystemExit(f"CSV is missing required column 'frame_label': {input_csv}")

        for row in reader:
            if row_contains_checkpoint(row):
                dropped_rows += 1
                continue
            kept_rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Rows kept: {len(kept_rows)}")
    print(f"Rows dropped: {dropped_rows}")
    return output_csv


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    input_csv = resolve_input_csv(input_path)
    clean_csv(input_csv, args.output_name)


if __name__ == "__main__":
    main()
