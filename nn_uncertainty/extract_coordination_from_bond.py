from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Zn-O coordination-like events from bond_change_detection CSV output "
            "and write a coordination_events.csv file."
        )
    )
    parser.add_argument(
        "bond_csv",
        help="Path to bond_change_detection CSV, typically bond_events.csv.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output CSV path. Defaults to coordination_events.csv in the same directory "
            "as the input bond CSV."
        ),
    )
    parser.add_argument(
        "--pair",
        default="O-Zn",
        help="Bond pair label to extract. Default: O-Zn",
    )
    return parser.parse_args()


def normalize_pair(value: str) -> str:
    text = (value or "").strip()
    if text == "Zn-O":
        return "O-Zn"
    return text


def extract_coordination_events(
    bond_csv: Path,
    output_csv: Path | None = None,
    pair: str = "O-Zn",
) -> Path:
    input_csv = bond_csv.expanduser().resolve()
    if not input_csv.exists():
        raise SystemExit(f"Input CSV does not exist: {input_csv}")

    resolved_output_csv = (
        output_csv.expanduser().resolve()
        if output_csv is not None
        else input_csv.with_name("coordination_events.csv")
    )
    target_pair = normalize_pair(pair)

    rows_by_frame: dict[str, dict[str, object]] = {}
    counts_by_frame: dict[str, dict[str, int]] = defaultdict(lambda: {"formed": 0, "broken": 0, "total": 0})

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"frame_label", "pair"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise SystemExit(
                f"Input CSV is missing required columns: {', '.join(missing)}"
            )

        for row in reader:
            current_pair = normalize_pair(row.get("pair", ""))
            if current_pair != target_pair:
                continue

            frame_label = row.get("frame_label", "").strip()
            if not frame_label:
                continue

            previous_frame_label = row.get("previous_frame_label", "").strip()
            event_type = (row.get("event_type", "") or "").strip().lower()

            if frame_label not in rows_by_frame:
                rows_by_frame[frame_label] = {
                    "frame_index": row.get("frame_index", ""),
                    "frame_label": frame_label,
                    "previous_frame_label": previous_frame_label,
                    "coordination_event_count": 0,
                    "formed_count": 0,
                    "broken_count": 0,
                    "pair": target_pair,
                }

            counts = counts_by_frame[frame_label]
            counts["total"] += 1
            if event_type == "formed":
                counts["formed"] += 1
            elif event_type == "broken":
                counts["broken"] += 1

    ordered_rows = sorted(
        rows_by_frame.values(),
        key=lambda row: int(float(row["frame_index"])) if str(row["frame_index"]).strip() else 10**18,
    )

    for row in ordered_rows:
        counts = counts_by_frame[str(row["frame_label"])]
        row["coordination_event_count"] = counts["total"]
        row["formed_count"] = counts["formed"]
        row["broken_count"] = counts["broken"]

    resolved_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "frame_label",
                "previous_frame_label",
                "coordination_event_count",
                "formed_count",
                "broken_count",
                "pair",
            ],
        )
        writer.writeheader()
        writer.writerows(ordered_rows)

    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {resolved_output_csv}")
    print(f"Extracted frames: {len(ordered_rows)}")
    print(f"Target pair: {target_pair}")
    return resolved_output_csv


def main() -> None:
    args = parse_args()
    extract_coordination_events(
        bond_csv=Path(args.bond_csv),
        output_csv=Path(args.output) if args.output else None,
        pair=args.pair,
    )


if __name__ == "__main__":
    main()
