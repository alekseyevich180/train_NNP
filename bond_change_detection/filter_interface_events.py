from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EVENTS_COLUMN = (
    "events(interface_type,event_type,atom_i,atom_j,group_i,group_j,"
    "symbol_i,symbol_j,pair,distance_broken,distance_formed)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter grouped interface events exported by bond_change_detection/run.py."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/bond_events_interfaces.csv"),
        help="Grouped interface events CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/bond_events_filtered.csv"),
        help="Filtered flat event CSV.",
    )
    parser.add_argument("--interface-type", default="organic-surface")
    parser.add_argument("--group-a", default="organic")
    parser.add_argument("--symbol-a", default="H")
    parser.add_argument("--group-b", default="surface")
    parser.add_argument("--symbol-b", default="O")
    return parser.parse_args()


def iter_grouped_events(row: dict[str, str]):
    text = row.get(EVENTS_COLUMN, "") or ""
    for payload in re.findall(r"\{([^}]*)\}", text):
        parts = payload.split(",")
        if len(parts) != 11:
            continue
        yield {
            "frame_index": row["frame_index"],
            "frame_label": row["frame_label"],
            "previous_frame_label": row["previous_frame_label"],
            "interface_type": parts[0],
            "event_type": parts[1],
            "atom_i": parts[2],
            "atom_j": parts[3],
            "group_i": parts[4],
            "group_j": parts[5],
            "symbol_i": parts[6],
            "symbol_j": parts[7],
            "pair": parts[8],
            "distance_broken": parts[9],
            "distance_formed": parts[10],
        }


def matches(event: dict[str, str], args: argparse.Namespace) -> bool:
    if event["interface_type"] != args.interface_type:
        return False

    forward = (
        event["group_i"] == args.group_a
        and event["symbol_i"] == args.symbol_a
        and event["group_j"] == args.group_b
        and event["symbol_j"] == args.symbol_b
    )
    reverse = (
        event["group_i"] == args.group_b
        and event["symbol_i"] == args.symbol_b
        and event["group_j"] == args.group_a
        and event["symbol_j"] == args.symbol_a
    )
    return forward or reverse


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_index",
        "frame_label",
        "previous_frame_label",
        "interface_type",
        "event_type",
        "atom_i",
        "atom_j",
        "group_i",
        "group_j",
        "symbol_i",
        "symbol_j",
        "pair",
        "distance_broken",
        "distance_formed",
    ]

    matched_count = 0
    with input_path.open("r", newline="", encoding="utf-8") as src, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            for event in iter_grouped_events(row):
                if matches(event, args):
                    writer.writerow(event)
                    matched_count += 1

    print(
        f"Filtered {matched_count} events from {input_path} to {output_path} "
        f"for {args.interface_type}: ({args.group_a},{args.symbol_a}) <-> ({args.group_b},{args.symbol_b})."
    )


if __name__ == "__main__":
    main()
