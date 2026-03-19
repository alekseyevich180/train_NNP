from __future__ import annotations

import argparse
import csv
from pathlib import Path


def normalize_row(row: dict[str, str | None]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        clean_key = key.lstrip("\ufeff").strip() if key is not None else ""
        normalized[clean_key] = "" if value is None else value
    return normalized


def parse_event_block(events_text: str) -> list[list[str]]:
    events: list[list[str]] = []
    for chunk in events_text.split(")"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("("):
            chunk = chunk[1:]
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) == 7:
            event_type, atom_i, atom_j, symbol_i, symbol_j, pair, distance_value = parts
            if event_type == "broken":
                parts = [
                    event_type,
                    atom_i,
                    atom_j,
                    symbol_i,
                    symbol_j,
                    pair,
                    distance_value,
                    "",
                ]
            elif event_type == "formed":
                parts = [
                    event_type,
                    atom_i,
                    atom_j,
                    symbol_i,
                    symbol_j,
                    pair,
                    "",
                    distance_value,
                ]
            else:
                continue
        if len(parts) != 8:
            continue
        events.append(parts)
    return events


def extract_pairs(input_csv: Path, output_csv: Path, target_pairs: set[str]) -> int:
    with input_csv.open("r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        fieldnames = [(name.lstrip("\ufeff").strip()) for name in (reader.fieldnames or [])]
        events_field = next(
            (name for name in fieldnames if name.startswith("events(")),
            "events",
        )

        with output_csv.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(
                outfile,
                fieldnames=[
                    "frame_index",
                    "frame_label",
                    "previous_frame_label",
                    "event_type",
                    "atom_i",
                    "atom_j",
                    "symbol_i",
                    "symbol_j",
                    "pair",
                    "distance_broken",
                    "distance_formed",
                ],
            )
            writer.writeheader()

            matched = 0
            for raw_row in reader:
                row = normalize_row(raw_row)
                for event in parse_event_block(row.get(events_field, "")):
                    (
                        event_type,
                        atom_i,
                        atom_j,
                        symbol_i,
                        symbol_j,
                        pair,
                        distance_broken,
                        distance_formed,
                    ) = event

                    if pair not in target_pairs:
                        continue

                    writer.writerow(
                        {
                            "frame_index": row.get("frame_index", ""),
                            "frame_label": row.get("frame_label", ""),
                            "previous_frame_label": row.get("previous_frame_label", ""),
                            "event_type": event_type,
                            "atom_i": atom_i,
                            "atom_j": atom_j,
                            "symbol_i": symbol_i,
                            "symbol_j": symbol_j,
                            "pair": pair,
                            "distance_broken": distance_broken,
                            "distance_formed": distance_formed,
                        }
                    )
                    matched += 1

    return matched


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Extract selected bond-pair events from bond_events.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "outputs" / "bond_events.csv",
        help="Path to the source bond_events.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "bond_events_C-C_C-O.csv",
        help="Path to the extracted event CSV.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["C-C", "C-O"],
        help="Bond pairs to keep. Default: C-C C-O",
    )
    args = parser.parse_args()

    input_csv = args.input.resolve()
    output_csv = args.output.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    matched = extract_pairs(input_csv, output_csv, set(args.pairs))
    print(f"Extracted {matched} events to {output_csv}")


if __name__ == "__main__":
    main()
