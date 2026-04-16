from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_AIMD_ROOT = Path(r"C:\Users\yingkaiwu\Desktop\Active-learning\NNP\aimd_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge DeepMD set folders from multiple deepmd_selected_from_nnp directories "
            "into one output directory with globally unique set names."
        )
    )
    parser.add_argument(
        "--aimd-root",
        type=Path,
        default=DEFAULT_AIMD_ROOT,
        help="Root directory that contains dataset folders. Default: %(default)s",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        help=(
            "Optional explicit deepmd_selected_from_nnp directories. "
            "If omitted, the script scans --aimd-root recursively."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AIMD_ROOT / "merged_deepmd_selected_from_nnp",
        help="Merged DeepMD output directory. Default: %(default)s",
    )
    parser.add_argument(
        "--summary-name",
        default="merged_sets_summary.csv",
        help="Summary CSV filename written into the output directory.",
    )
    parser.add_argument(
        "--copy-type-files",
        action="store_true",
        help="Copy top-level type.raw and type_map.raw from the first source directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory before writing merged results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be merged without copying files.",
    )
    return parser.parse_args()


def resolve_input_dirs(aimd_root: Path, inputs: list[Path] | None) -> list[Path]:
    if inputs:
        resolved = [path.expanduser().resolve() for path in inputs]
    else:
        resolved = sorted(
            path.resolve()
            for path in aimd_root.expanduser().resolve().rglob("deepmd_selected_from_nnp")
            if path.is_dir()
        )

    unique_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path in seen:
            continue
        if not path.exists():
            raise SystemExit(f"Input directory not found: {path}")
        unique_dirs.append(path)
        seen.add(path)
    return unique_dirs


def collect_set_dirs(input_dirs: list[Path]) -> list[tuple[Path, Path]]:
    collected: list[tuple[Path, Path]] = []
    for source_dir in input_dirs:
        set_dirs = sorted(
            path for path in source_dir.iterdir() if path.is_dir() and path.name.startswith("set.")
        )
        for set_dir in set_dirs:
            collected.append((source_dir, set_dir))
    return collected


def prepare_output_dir(output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not overwrite:
                raise SystemExit(
                    f"Output directory is not empty: {output_dir}\n"
                    "Use --overwrite to replace it."
                )
            if not dry_run:
                shutil.rmtree(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)


def copy_type_files(first_source_dir: Path, output_dir: Path, dry_run: bool) -> list[str]:
    copied: list[str] = []
    for filename in ("type.raw", "type_map.raw"):
        source_file = first_source_dir / filename
        if source_file.exists():
            copied.append(filename)
            if not dry_run:
                shutil.copy2(source_file, output_dir / filename)
    return copied


def write_summary(summary_csv: Path, rows: list[dict[str, str]], dry_run: bool) -> None:
    if dry_run:
        return
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "new_set_name",
                "source_dataset",
                "source_deepmd_dir",
                "original_set_name",
                "original_set_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    aimd_root = args.aimd_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_dirs = resolve_input_dirs(aimd_root, args.inputs)

    if not input_dirs:
        raise SystemExit(
            f"No deepmd_selected_from_nnp directories found under {aimd_root}"
        )

    merged_sets = collect_set_dirs(input_dirs)
    if not merged_sets:
        raise SystemExit("No set.* directories were found in the input directories.")

    prepare_output_dir(output_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    width = max(3, len(str(len(merged_sets) - 1)))
    summary_rows: list[dict[str, str]] = []

    copied_type_files: list[str] = []
    if args.copy_type_files:
        copied_type_files = copy_type_files(input_dirs[0], output_dir, dry_run=args.dry_run)

    for index, (source_dir, set_dir) in enumerate(merged_sets):
        new_set_name = f"set.{index:0{width}d}"
        destination_dir = output_dir / new_set_name
        dataset_name = source_dir.parent.name
        summary_rows.append(
            {
                "new_set_name": new_set_name,
                "source_dataset": dataset_name,
                "source_deepmd_dir": str(source_dir),
                "original_set_name": set_dir.name,
                "original_set_path": str(set_dir),
            }
        )
        if not args.dry_run:
            shutil.copytree(set_dir, destination_dir)

    summary_csv = output_dir / args.summary_name
    write_summary(summary_csv, summary_rows, dry_run=args.dry_run)

    print(f"AIMD root: {aimd_root}")
    print("Input directories:")
    for source_dir in input_dirs:
        print(f"  - {source_dir}")
    print(f"Total source directories: {len(input_dirs)}")
    print(f"Total copied sets: {len(merged_sets)}")
    print(f"Output directory: {output_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Copied type files: {', '.join(copied_type_files) if copied_type_files else 'none'}")
    print(f"Dry run: {'yes' if args.dry_run else 'no'}")


if __name__ == "__main__":
    main()
