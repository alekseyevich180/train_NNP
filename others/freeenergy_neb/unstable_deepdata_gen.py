from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator


@dataclass(frozen=True)
class ScreenConfig:
    input_files: tuple[str, ...] = ("neb_state10.cif",)
    input_glob: str = "*.cif"
    calc_mode: str = "CRYSTAL"
    fix_bottom: bool = True
    fix_z_max: float = 1.0
    relax_fmax: float = 0.05
    trajectory_steps: int = 300
    relax_maxstep: float = 0.2
    stable_window_start: int = 100
    selected_frames: int = 100
    final_relax_fmax: float = 0.05
    final_relax_steps: int = 50
    deepmd_dirname: str = "deepmd_dataset"
    min_samples: int = 100


CONFIG = ScreenConfig()


def build_calculator(calc_mode: str = "CRYSTAL") -> ASECalculator:
    estimator = Estimator(calc_mode=calc_mode)
    return ASECalculator(estimator)


def fixed_bottom_layer(atoms: Atoms, z_max: float = 1.0) -> FixAtoms:
    return FixAtoms(indices=[atom.index for atom in atoms if atom.position[2] <= z_max])


CALCULATOR = build_calculator(CONFIG.calc_mode)


class DeepMDWriter:
    def __init__(self, atoms: Atoms, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.write_type_files(atoms)

    def write_type_files(self, atoms: Atoms) -> None:
        symbols = atoms.get_chemical_symbols()
        uniq = sorted(set(symbols))
        type_map = {symbol: idx for idx, symbol in enumerate(uniq)}
        type_list = np.array([type_map[symbol] for symbol in symbols], dtype=int)

        np.savetxt(self.root / "type.raw", type_list, fmt="%d")
        (self.root / "type_map.raw").write_text("\n".join(uniq) + "\n", encoding="utf-8")

    def add_frame(self, atoms: Atoms, step_id: int) -> None:
        set_dir = self.root / f"set.{step_id}"
        set_dir.mkdir(parents=True, exist_ok=True)

        np.save(set_dir / "coord.npy", np.array([atoms.get_positions().reshape(-1)]))
        np.save(set_dir / "force.npy", np.array([atoms.get_forces().reshape(-1)]))
        np.save(set_dir / "energy.npy", np.array([atoms.get_potential_energy()]))
        np.save(set_dir / "box.npy", np.array([atoms.get_cell().array.reshape(-1)]))


def _normalize_input_names(input_names: Sequence[str] | None) -> list[str]:
    if not input_names:
        return []

    normalized: list[str] = []
    for raw_name in input_names:
        if raw_name is None:
            continue

        name = str(raw_name).strip()
        if not name:
            continue

        if name.startswith("[") and name.endswith("]"):
            name = name[1:-1].strip().strip("'\"")
        if not name:
            continue

        normalized.append(name)

    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relax CIF structures, record short BFGS trajectories, select relatively stable frames, "
            "and export them as DeepMD data."
        )
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=list(CONFIG.input_files),
        help="Explicit CIF filenames in the current directory.",
    )
    parser.add_argument(
        "--input-glob",
        default=CONFIG.input_glob,
        help="Glob pattern for input structures in the current directory.",
    )
    parser.add_argument(
        "--output-root",
        default="screening_output",
        help="Directory used to store relaxed structures, recorded trajectories, DeepMD data, and summaries.",
    )
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=CONFIG.trajectory_steps,
        help="Number of BFGS relaxation steps to record per structure.",
    )
    parser.add_argument(
        "--stable-window-start",
        type=int,
        default=CONFIG.stable_window_start,
        help="Start selecting stable frames from this recorded step.",
    )
    parser.add_argument(
        "--selected-frames",
        type=int,
        default=CONFIG.selected_frames,
        help="Number of frames to select per structure for the DeepMD dataset.",
    )
    parser.add_argument(
        "--relax-fmax",
        type=float,
        default=CONFIG.relax_fmax,
        help="Force threshold for the initial structure relaxation and trajectory run.",
    )
    parser.add_argument(
        "--relax-maxstep",
        type=float,
        default=CONFIG.relax_maxstep,
        help="Maximum atomic displacement per BFGS step during trajectory sampling.",
    )
    parser.add_argument(
        "--final-relax-fmax",
        type=float,
        default=CONFIG.final_relax_fmax,
        help="Force threshold for the short final relaxation applied to selected frames.",
    )
    parser.add_argument(
        "--final-relax-steps",
        type=int,
        default=CONFIG.final_relax_steps,
        help="Maximum number of BFGS steps for the short final relaxation of selected frames.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=CONFIG.min_samples,
        help="Required minimum number of selected frames per structure.",
    )
    args, _unknown = parser.parse_known_args(argv)
    args.input_files = _normalize_input_names(args.input_files)
    return args


def expected_recorded_frame_count(trajectory_steps: int) -> int:
    return trajectory_steps + 1


def resolve_input_paths(cwd: Path, args: argparse.Namespace) -> list[Path]:
    if args.input_files:
        input_paths = []
        for name in args.input_files:
            if any(char in name for char in "\r\n\t"):
                raise ValueError(
                    "Input filename contains control characters. "
                    f"Received {name!r}. In Jupyter, pass --input-files as separate filenames."
                )
            path = (cwd / name).resolve()
            if not path.exists():
                raise ValueError(
                    "Input structure not found: "
                    f"{path}. Received input_files={args.input_files!r}"
                )
            if not path.is_file():
                raise ValueError(f"Input path is not a file: {path}")
            input_paths.append(path)
        return sorted(input_paths)

    return sorted(path for path in cwd.glob(args.input_glob) if path.is_file() and path.parent == cwd)


def load_structure(path: Path) -> Atoms:
    atoms = read(str(path))
    print(f"Loaded {path.name}: {len(atoms)} atoms")
    return atoms


def apply_constraints(atoms: Atoms, config: ScreenConfig) -> None:
    if config.fix_bottom:
        atoms.set_constraint(fixed_bottom_layer(atoms, z_max=config.fix_z_max))
    else:
        atoms.set_constraint()


def compute_energy(atoms: Atoms, config: ScreenConfig) -> float:
    atoms.calc = CALCULATOR
    apply_constraints(atoms, config)
    return float(atoms.get_potential_energy())


def relax_structure(atoms: Atoms, config: ScreenConfig, trajectory_path: Path | None = None) -> Atoms:
    relaxed = atoms.copy()
    relaxed.calc = CALCULATOR
    apply_constraints(relaxed, config)
    BFGS(relaxed, trajectory=str(trajectory_path) if trajectory_path else None, logfile=None).run(
        fmax=config.relax_fmax
    )
    return relaxed


def _record_frame(atoms: Atoms, config: ScreenConfig, sampled_dir: Path, step: int) -> dict[str, object]:
    frame = atoms.copy()
    frame.calc = CALCULATOR
    apply_constraints(frame, config)

    energy = float(frame.get_potential_energy())
    forces = frame.get_forces()
    force_norms = np.linalg.norm(forces, axis=1)
    max_force = float(np.max(force_norms))
    rms_force = float(np.sqrt(np.mean(force_norms**2)))

    filename = f"relax_step_{step:04d}.cif"
    write(str(sampled_dir / filename), frame)
    return {
        "step": step,
        "frame": frame,
        "energy_eV": energy,
        "max_force_eVA": max_force,
        "rms_force_eVA": rms_force,
        "file": filename,
    }


def run_relaxation_presampling(atoms: Atoms, config: ScreenConfig, sampled_dir: Path) -> pd.DataFrame:
    sampled_dir.mkdir(parents=True, exist_ok=True)

    work_atoms = atoms.copy()
    work_atoms.calc = CALCULATOR
    apply_constraints(work_atoms, config)

    rows: list[dict[str, object]] = []
    rows.append(_record_frame(work_atoms, config, sampled_dir, step=0))

    optimizer = BFGS(work_atoms, logfile=None, maxstep=config.relax_maxstep)

    def record_current_step() -> None:
        rows.append(_record_frame(work_atoms, config, sampled_dir, step=len(rows)))

    optimizer.attach(record_current_step, interval=1)
    optimizer.run(fmax=config.relax_fmax, steps=config.trajectory_steps)

    trajectory_df = pd.DataFrame(
        [
            {
                "step": row["step"],
                "energy_eV": row["energy_eV"],
                "max_force_eVA": row["max_force_eVA"],
                "rms_force_eVA": row["rms_force_eVA"],
                "file": row["file"],
            }
            for row in rows
        ]
    )
    trajectory_df.to_csv(sampled_dir / "relaxation_sampling.csv", index=False)
    trajectory_df.attrs["frames"] = rows
    return trajectory_df


def select_stable_frames(
    trajectory_df: pd.DataFrame,
    config: ScreenConfig,
    optimized_dir: Path,
) -> tuple[list[Atoms], pd.DataFrame]:
    if trajectory_df.empty:
        raise ValueError("No relaxation frames were recorded.")

    optimized_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, object]] = trajectory_df.attrs.get("frames", [])
    if not frames:
        raise ValueError("Relaxation frame metadata is missing.")

    candidate_df = trajectory_df[trajectory_df["step"] >= config.stable_window_start].copy()
    if candidate_df.empty:
        candidate_df = trajectory_df.copy()

    candidate_df = candidate_df.sort_values(["max_force_eVA", "energy_eV", "step"]).reset_index(drop=True)
    keep_count = min(config.selected_frames, len(candidate_df))
    if keep_count < config.min_samples:
        raise ValueError(
            f"Only {keep_count} candidate frames available, below min_samples={config.min_samples}."
        )

    if keep_count < len(candidate_df):
        selected_positions = np.linspace(0, len(candidate_df) - 1, keep_count, dtype=int)
        selected_df = candidate_df.iloc[selected_positions].copy()
    else:
        selected_df = candidate_df.copy()

    selected_df = selected_df.sort_values("step").reset_index(drop=True)
    frames_by_step = {int(row["step"]): row["frame"] for row in frames}

    selected_atoms: list[Atoms] = []
    selected_rows: list[dict[str, object]] = []
    for selected_idx, row in selected_df.iterrows():
        step = int(row["step"])
        candidate = frames_by_step[step].copy()
        candidate.calc = CALCULATOR
        apply_constraints(candidate, config)
        BFGS(candidate, logfile=None).run(
            fmax=config.final_relax_fmax,
            steps=config.final_relax_steps,
        )
        energy = float(candidate.get_potential_energy())

        filename = f"selected_{selected_idx:04d}_step_{step:04d}.cif"
        write(str(optimized_dir / filename), candidate)

        selected_atoms.append(candidate)
        selected_rows.append(
            {
                "selected_index": selected_idx,
                "source_step": step,
                "pre_energy_eV": float(row["energy_eV"]),
                "post_energy_eV": energy,
                "max_force_eVA": float(row["max_force_eVA"]),
                "rms_force_eVA": float(row["rms_force_eVA"]),
                "file": filename,
            }
        )

    selected_summary = pd.DataFrame(selected_rows)
    selected_summary.to_csv(optimized_dir / "selected_frames.csv", index=False)
    return selected_atoms, selected_summary


def export_deepmd_frames(selected_atoms: list[Atoms], deepmd_root: Path) -> None:
    if not selected_atoms:
        raise ValueError("No selected frames available for DeepMD export.")

    deepmd_writer = DeepMDWriter(selected_atoms[0], deepmd_root)
    for step_id, atoms in enumerate(selected_atoms):
        deepmd_writer.add_frame(atoms, step_id)


def screen_structure(input_path: Path, config: ScreenConfig, output_root: Path) -> dict[str, object]:
    structure_name = input_path.stem
    relaxed_dir = output_root / "relaxed"
    sampled_dir = output_root / "relaxation_traj" / structure_name
    optimized_dir = output_root / "selected_frames" / structure_name
    selected_dir = output_root / "selected"
    deepmd_dir = output_root / config.deepmd_dirname / structure_name
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    atoms = load_structure(input_path)
    initial_energy = compute_energy(atoms.copy(), config)

    relaxed = relax_structure(
        atoms,
        config,
        trajectory_path=relaxed_dir / f"{structure_name}_relax.traj",
    )
    relaxed_energy = float(relaxed.get_potential_energy())
    relaxed_path = relaxed_dir / f"{structure_name}_relaxed.cif"
    write(str(relaxed_path), relaxed)

    trajectory_df = run_relaxation_presampling(relaxed, config, sampled_dir)
    selected_atoms, selected_summary = select_stable_frames(trajectory_df, config, optimized_dir)
    export_deepmd_frames(selected_atoms, deepmd_dir)

    best_row = selected_summary.loc[selected_summary["post_energy_eV"].idxmin()]
    best_atoms = selected_atoms[int(best_row["selected_index"])].copy()
    best_energy = float(best_row["post_energy_eV"])

    selected_path = selected_dir / f"{structure_name}_selected.cif"
    write(str(selected_path), best_atoms)
    print(
        f"{input_path.name}: initial={initial_energy:.6f} eV, "
        f"relaxed={relaxed_energy:.6f} eV, selected={best_energy:.6f} eV"
    )

    return {
        "input_file": input_path.name,
        "initial_energy_eV": initial_energy,
        "relaxed_energy_eV": relaxed_energy,
        "selected_energy_eV": best_energy,
        "relaxed_file": str(relaxed_path),
        "selected_file": str(selected_path),
        "deepmd_root": str(deepmd_dir),
        "sample_count": len(selected_atoms),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cwd = Path.cwd()
    output_root = cwd / args.output_root
    input_paths = resolve_input_paths(cwd, args)

    if not input_paths:
        raise SystemExit(f"No input structures found in {cwd}")

    recorded_frames = expected_recorded_frame_count(args.trajectory_steps)
    projected_samples = min(args.selected_frames, max(0, recorded_frames - args.stable_window_start))
    if projected_samples < args.min_samples:
        raise SystemExit(
            "Sampling setup is too small: "
            f"trajectory_steps={args.trajectory_steps}, stable_window_start={args.stable_window_start}, "
            f"selected_frames={args.selected_frames}, expected_samples={projected_samples}, "
            f"required_min_samples={args.min_samples}. Increase --trajectory-steps or lower --min-samples."
        )

    print(
        f"Sampling plan: {recorded_frames} recorded relaxation frames, "
        f"{projected_samples} selected frames per structure "
        f"(stable_window_start={args.stable_window_start})"
    )

    runtime_config = ScreenConfig(
        calc_mode=CONFIG.calc_mode,
        fix_bottom=CONFIG.fix_bottom,
        fix_z_max=CONFIG.fix_z_max,
        relax_fmax=args.relax_fmax,
        trajectory_steps=args.trajectory_steps,
        relax_maxstep=args.relax_maxstep,
        stable_window_start=args.stable_window_start,
        selected_frames=args.selected_frames,
        final_relax_fmax=args.final_relax_fmax,
        final_relax_steps=args.final_relax_steps,
        deepmd_dirname=CONFIG.deepmd_dirname,
        min_samples=args.min_samples,
    )

    summary_rows = [screen_structure(input_path, runtime_config, output_root) for input_path in input_paths]
    summary_csv = output_root / "screening_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_file",
                "initial_energy_eV",
                "relaxed_energy_eV",
                "selected_energy_eV",
                "relaxed_file",
                "selected_file",
                "deepmd_root",
                "sample_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved screening summary: {summary_csv}")


if __name__ == "__main__":
    main()
