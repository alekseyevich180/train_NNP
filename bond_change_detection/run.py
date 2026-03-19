from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase.io import iread, read, write
from ase.neighborlist import neighbor_list

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency issue is environment-specific
    raise SystemExit("PyYAML is required to run bond_change_detection.") from exc


@dataclass(frozen=True)
class BondRecord:
    atom_i: int
    atom_j: int
    symbol_i: str
    symbol_j: str
    pair: str
    distance: float


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def normalize_pair(symbol_a: str, symbol_b: str) -> tuple[str, str]:
    return tuple(sorted((symbol_a, symbol_b)))


def parse_cutoffs(raw_cutoffs: dict[str, float]) -> dict[tuple[str, str], float]:
    parsed: dict[tuple[str, str], float] = {}
    for pair_label, cutoff in raw_cutoffs.items():
        left, right = pair_label.split("-", maxsplit=1)
        parsed[normalize_pair(left, right)] = float(cutoff)
    return parsed


def frame_label_from_path(input_path: Path, frame_file: Path) -> str:
    relative_path = frame_file.relative_to(input_path)
    return str(relative_path.with_suffix("")).replace("\\", "__").replace("/", "__")


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


def discover_frame_paths(input_path: Path, input_format: str, selection: dict[str, Any]) -> list[Path]:
    if input_format == "cif_dir":
        if not input_path.exists():
            return []
        return select_entries(sorted(input_path.rglob("*.cif")), selection)
    raise ValueError(f"Frame paths are only supported for cif_dir input, got: {input_format}")


def iter_frames(input_path: Path, input_format: str, selection: dict[str, Any]):
    if input_format == "cif_dir":
        for frame_file in discover_frame_paths(input_path, input_format, selection):
            yield frame_label_from_path(input_path, frame_file), read(frame_file)
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
            yield f"frame_{idx:08d}", atoms
            selected_count += 1
        return
    raise ValueError(f"Unsupported input format: {input_format}")


def build_bond_map(atoms: Any, cutoffs: dict[tuple[str, str], float]) -> dict[tuple[int, int], BondRecord]:
    neighbor_cutoffs = {(left, right): cutoff for (left, right), cutoff in cutoffs.items()}
    indices_i, indices_j, distances = neighbor_list("ijd", atoms, neighbor_cutoffs)

    bond_map: dict[tuple[int, int], BondRecord] = {}
    symbols = atoms.get_chemical_symbols()
    for atom_i, atom_j, distance in zip(indices_i, indices_j, distances):
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if atom_i >= atom_j:
            continue
        symbol_i = symbols[atom_i]
        symbol_j = symbols[atom_j]
        pair_key = normalize_pair(symbol_i, symbol_j)
        if pair_key not in cutoffs:
            continue
        pair_label = f"{pair_key[0]}-{pair_key[1]}"
        bond_map[(atom_i, atom_j)] = BondRecord(
            atom_i=atom_i,
            atom_j=atom_j,
            symbol_i=symbol_i,
            symbol_j=symbol_j,
            pair=pair_label,
            distance=float(distance),
        )
    return bond_map


def export_events(events: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "frame_index",
        "frame_label",
        "previous_frame_label",
        "event_count",
        "events(event_type,atom_i,atom_j,symbol_i,symbol_j,pair,distance_broken,distance_formed)",
    ]
    events_field = fieldnames[-1]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        grouped_rows: list[dict[str, Any]] = []
        current_row: dict[str, Any] | None = None
        current_events: list[str] = []

        for event in events:
            row_key = (
                event["frame_index"],
                event["frame_label"],
                event["previous_frame_label"],
            )
            current_key = None
            if current_row is not None:
                current_key = (
                    current_row["frame_index"],
                    current_row["frame_label"],
                    current_row["previous_frame_label"],
                )

            if current_row is None or row_key != current_key:
                if current_row is not None:
                    current_row["event_count"] = len(current_events)
                    current_row["events"] = " ".join(current_events)
                    grouped_rows.append(current_row)
                current_row = {
                    "frame_index": event["frame_index"],
                    "frame_label": event["frame_label"],
                    "previous_frame_label": event["previous_frame_label"],
                    "event_count": 0,
                    events_field: "",
                }
                current_events = []

            current_events.append(
                f"({event['event_type']},{event['atom_i']},{event['atom_j']},"
                f"{event['symbol_i']},{event['symbol_j']},{event['pair']},"
                f"{event['distance_previous']},{event['distance_current']})"
            )

        if current_row is not None:
            current_row["event_count"] = len(current_events)
            current_row[events_field] = " ".join(current_events)
            grouped_rows.append(current_row)

        writer.writerows(grouped_rows)


def export_pair_events(events: list[dict[str, Any]], csv_path: Path, pair_label: str) -> int:
    fieldnames = [
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
    ]
    matched_events = [event for event in events if event["pair"] == pair_label]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in matched_events:
            writer.writerow(
                {
                    "frame_index": event["frame_index"],
                    "frame_label": event["frame_label"],
                    "previous_frame_label": event["previous_frame_label"],
                    "event_type": event["event_type"],
                    "atom_i": event["atom_i"],
                    "atom_j": event["atom_j"],
                    "symbol_i": event["symbol_i"],
                    "symbol_j": event["symbol_j"],
                    "pair": event["pair"],
                    "distance_broken": event["distance_previous"],
                    "distance_formed": event["distance_current"],
                }
            )

    return len(matched_events)


def export_key_frame(frame_label: str, atoms: Any, output_dir: Path) -> None:
    output_path = output_dir / f"{frame_label}.cif"
    write(output_path, atoms)


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")

    output_root = root / config["output"]["root"]
    output_root.mkdir(exist_ok=True)

    input_path = (root / config["input"]["trajectory"]).resolve()
    input_format = config["input"]["format"]
    cutoffs = parse_cutoffs(config["chemistry"]["cutoffs"])
    selection = config.get("selection", {})
    parsed_selection = parse_selection(selection)
    save_changed_frames = bool(selection.get("save_changed_frames", False))
    progress_interval = max(1, int(selection.get("progress_interval", 100)))
    key_frames_dir = output_root / config["output"]["key_frames_dir"]
    if save_changed_frames:
        key_frames_dir.mkdir(exist_ok=True)

    events_csv = output_root / config["output"]["events_csv"]
    cc_events_csv = output_root / "bond_events_C-C.csv"
    co_events_csv = output_root / "bond_events_C-O.csv"
    summary_json = output_root / config["output"]["summary_json"]
    frame_iter = iter_frames(input_path, input_format, selection)

    try:
        previous_label, previous_atoms = next(frame_iter)
    except StopIteration:
        export_events([], events_csv)
        export_pair_events([], cc_events_csv, "C-C")
        export_pair_events([], co_events_csv, "C-O")
        summary = {
            "module": "bond_change_detection",
            "status": "no_input",
            "input_path": str(input_path),
            "input_format": input_format,
            "frames_loaded": 0,
            "events_detected": 0,
            "message": "Provide at least two trajectory frames to detect bond changes.",
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("No bond changes computed. Frames loaded: 0.")
        return

    previous_bonds = build_bond_map(previous_atoms, cutoffs)
    events: list[dict[str, Any]] = []
    key_frame_labels: set[str] = set()
    frames_loaded = 1

    if input_format == "cif_dir":
        total_frames = len(discover_frame_paths(input_path, input_format, selection))
    else:
        total_frames = None

    for frame_index, (frame_label, atoms) in enumerate(frame_iter, start=1):
        frames_loaded += 1
        if frame_index % progress_interval == 0:
            if total_frames is None:
                print(f"Processed {frame_index + 1} frames...")
            else:
                print(f"Processed {frame_index + 1}/{total_frames} frames...")

        current_bonds = build_bond_map(atoms, cutoffs)

        formed_keys = sorted(set(current_bonds) - set(previous_bonds))
        broken_keys = sorted(set(previous_bonds) - set(current_bonds))

        for bond_key in formed_keys:
            bond = current_bonds[bond_key]
            events.append(
                {
                    "frame_index": frame_index,
                    "frame_label": frame_label,
                    "previous_frame_label": previous_label,
                    "event_type": "formed",
                    "atom_i": bond.atom_i,
                    "atom_j": bond.atom_j,
                    "symbol_i": bond.symbol_i,
                    "symbol_j": bond.symbol_j,
                    "pair": bond.pair,
                    "distance_previous": "",
                    "distance_current": f"{bond.distance:.6f}",
                }
            )
            key_frame_labels.add(frame_label)

        for bond_key in broken_keys:
            bond = previous_bonds[bond_key]
            events.append(
                {
                    "frame_index": frame_index,
                    "frame_label": frame_label,
                    "previous_frame_label": previous_label,
                    "event_type": "broken",
                    "atom_i": bond.atom_i,
                    "atom_j": bond.atom_j,
                    "symbol_i": bond.symbol_i,
                    "symbol_j": bond.symbol_j,
                    "pair": bond.pair,
                    "distance_previous": f"{bond.distance:.6f}",
                    "distance_current": "",
                }
            )
            key_frame_labels.add(frame_label)

        if save_changed_frames and frame_label in key_frame_labels:
            export_key_frame(frame_label, atoms, key_frames_dir)

        previous_label = frame_label
        previous_bonds = current_bonds

    if frames_loaded < 2:
        export_events([], events_csv)
        export_pair_events([], cc_events_csv, "C-C")
        export_pair_events([], co_events_csv, "C-O")
        summary = {
            "module": "bond_change_detection",
            "status": "insufficient_frames",
            "input_path": str(input_path),
            "input_format": input_format,
            "frames_loaded": frames_loaded,
            "events_detected": 0,
            "message": "Provide at least two trajectory frames to detect bond changes.",
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"No bond changes computed. Frames loaded: {frames_loaded}.")
        return

    export_events(events, events_csv)
    cc_event_count = export_pair_events(events, cc_events_csv, "C-C")
    co_event_count = export_pair_events(events, co_events_csv, "C-O")

    summary = {
        "module": "bond_change_detection",
        "status": "completed",
        "input_path": str(input_path),
        "input_format": input_format,
        "frames_loaded": frames_loaded,
        "events_detected": len(events),
        "frames_with_changes": len(key_frame_labels),
        "pair_cutoffs": {f"{left}-{right}": value for (left, right), value in cutoffs.items()},
        "saved_key_frames": save_changed_frames,
        "exported_pair_event_files": {
            "C-C": {"path": str(cc_events_csv), "events": cc_event_count},
            "C-O": {"path": str(co_events_csv), "events": co_event_count},
        },
        "progress_interval": progress_interval,
        "frame_selection": parsed_selection,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "bond_change_detection completed: "
        f"{len(events)} events across {len(key_frame_labels)} changed frames. "
        f"C-C: {cc_event_count}, C-O: {co_event_count}."
    )


if __name__ == "__main__":
    main()
