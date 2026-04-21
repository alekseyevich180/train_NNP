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
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.units import fs
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator


@dataclass(frozen=True)
class ScreenConfig:
    input_files: tuple[str, ...] = ("ketone.cif",)
    input_glob: str = "*.cif"
    calc_mode: str = "CRYSTAL"
    fix_bottom: bool = True
    fix_z_max: float = 1.0
    relax_fmax: float = 0.05
    temperature_K: float = 300.0
    aimd_steps: int = 20000
    timestep_fs: float = 0.5
    tau_t: float = 100.0
    sample_interval: int = 100
    relax_sampled_frames: bool = True
    sampled_frame_fmax: float = 0.05
    deepmd_dirname: str = "deepmd_dataset"
    min_samples: int = 200


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

        # Notebook calls sometimes pass a single stringified list or include newline-only tokens.
        if name.startswith("[") and name.endswith("]"):
            name = name[1:-1].strip().strip("'\"")
        if not name:
            continue

        normalized.append(name)

    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relax CIF structures in the current folder, run short AIMD sampling, "
            "export sampled frames as DeepMD data, and select the lowest-energy sampled structure."
        )
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=list(CONFIG.input_files),
        help=(
            "Explicit CIF filenames in the current directory. "
            "Example: --input-files a.cif b.cif c.cif"
        ),
    )
    parser.add_argument(
        "--input-glob",
        default=CONFIG.input_glob,
        help=(
            "Glob pattern for input structures in the current directory. "
            "Ignored when --input-files is provided. Default: *.cif"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="screening_output",
        help="Directory used to store relaxed structures, AIMD samples, DeepMD data, and summaries.",
    )
    parser.add_argument(
        "--aimd-steps",
        type=int,
        default=CONFIG.aimd_steps,
        help="Number of NVT AIMD steps per structure.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=CONFIG.sample_interval,
        help="Write one sampled frame every N MD steps.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=CONFIG.temperature_K,
        help="AIMD temperature in Kelvin.",
    )
    parser.add_argument(
        "--relax-fmax",
        type=float,
        default=CONFIG.relax_fmax,
        help="Force threshold for initial structural relaxation.",
    )
    parser.add_argument(
        "--sampled-frame-fmax",
        type=float,
        default=CONFIG.sampled_frame_fmax,
        help="Force threshold for relaxing sampled AIMD frames.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=CONFIG.min_samples,
        help="Required minimum number of sampled AIMD frames per structure.",
    )
    args, _unknown = parser.parse_known_args(argv)
    args.input_files = _normalize_input_names(args.input_files)
    return args


def expected_sample_count(aimd_steps: int, sample_interval: int) -> int:
    return aimd_steps // sample_interval


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


def run_aimd_presampling(
    atoms: Atoms,
    config: ScreenConfig,
    sampled_dir: Path,
    deepmd_root: Path,
) -> list[Atoms]:
    sampled_dir.mkdir(parents=True, exist_ok=True)

    md_atoms = atoms.copy()
    md_atoms.calc = CALCULATOR
    apply_constraints(md_atoms, config)

    deepmd_writer = DeepMDWriter(md_atoms, deepmd_root)

    MaxwellBoltzmannDistribution(md_atoms, temperature_K=config.temperature_K)
    dyn = NVTBerendsen(
        md_atoms,
        timestep=config.timestep_fs * fs,
        temperature_K=config.temperature_K,
        taut=config.tau_t,
    )

    sampled_frames: list[Atoms] = []
    sampled_rows: list[dict[str, object]] = []
    step_counter = {"step": 0}

    def sample_frame() -> None:
        step_counter["step"] += 1
        if step_counter["step"] % config.sample_interval != 0:
            return

        frame = md_atoms.copy()
        frame.calc = CALCULATOR
        apply_constraints(frame, config)
        energy = float(frame.get_potential_energy())

        sampled_frames.append(frame)
        filename = f"step_{step_counter['step']:06d}.cif"
        write(str(sampled_dir / filename), frame)
        deepmd_writer.add_frame(frame, step_counter["step"])

        sampled_rows.append(
            {
                "step": step_counter["step"],
                "energy_eV": energy,
                "file": filename,
                "deepmd_set": f"set.{step_counter['step']}",
            }
        )

    dyn.attach(sample_frame, interval=1)
    dyn.run(config.aimd_steps)

    if sampled_rows:
        pd.DataFrame(sampled_rows).to_csv(sampled_dir / "energies.csv", index=False)
    return sampled_frames


def select_lowest_energy_sample(
    sampled_frames: list[Atoms],
    config: ScreenConfig,
    optimized_dir: Path,
) -> tuple[Atoms, float]:
    if not sampled_frames:
        raise ValueError("No AIMD frames were sampled.")

    optimized_dir.mkdir(parents=True, exist_ok=True)
    optimized_rows: list[dict[str, object]] = []
    best_atoms: Atoms | None = None
    best_energy: float | None = None

    for idx, frame in enumerate(sampled_frames):
        candidate = frame.copy()
        candidate.calc = CALCULATOR
        apply_constraints(candidate, config)
        if config.relax_sampled_frames:
            BFGS(candidate, logfile=None).run(fmax=config.sampled_frame_fmax)
        energy = float(candidate.get_potential_energy())
        filename = f"sample_{idx:04d}.cif"
        write(str(optimized_dir / filename), candidate)
        optimized_rows.append(
            {
                "sample_index": idx,
                "source_step": (idx + 1) * config.sample_interval,
                "energy_eV": energy,
                "file": filename,
            }
        )
        if best_energy is None or energy < best_energy:
            best_atoms = candidate.copy()
            best_energy = energy

    pd.DataFrame(optimized_rows).to_csv(optimized_dir / "energies.csv", index=False)
    assert best_atoms is not None and best_energy is not None
    return best_atoms, best_energy


def screen_structure(input_path: Path, config: ScreenConfig, output_root: Path) -> dict[str, object]:
    structure_name = input_path.stem
    relaxed_dir = output_root / "relaxed"
    sampled_dir = output_root / "aimd_samples" / structure_name
    optimized_dir = output_root / "optimized_samples" / structure_name
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

    sampled_frames = run_aimd_presampling(relaxed, config, sampled_dir, deepmd_dir)
    best_atoms, best_energy = select_lowest_energy_sample(sampled_frames, config, optimized_dir)

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
        "sample_count": len(sampled_frames),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cwd = Path.cwd()
    output_root = cwd / args.output_root
    input_paths = resolve_input_paths(cwd, args)

    if not input_paths:
        raise SystemExit(f"No input structures found in {cwd}")

    projected_samples = expected_sample_count(args.aimd_steps, args.sample_interval)
    if projected_samples < args.min_samples:
        raise SystemExit(
            "Sampling setup is too small: "
            f"aimd_steps={args.aimd_steps}, sample_interval={args.sample_interval}, "
            f"expected_samples={projected_samples}, required_min_samples={args.min_samples}. "
            "Increase --aimd-steps or decrease --sample-interval."
        )

    print(
        f"Sampling plan: {projected_samples} sampled AIMD frames per structure "
        f"(aimd_steps={args.aimd_steps}, sample_interval={args.sample_interval})"
    )

    runtime_config = ScreenConfig(
        calc_mode=CONFIG.calc_mode,
        fix_bottom=CONFIG.fix_bottom,
        fix_z_max=CONFIG.fix_z_max,
        relax_fmax=args.relax_fmax,
        temperature_K=args.temperature,
        aimd_steps=args.aimd_steps,
        timestep_fs=CONFIG.timestep_fs,
        tau_t=CONFIG.tau_t,
        sample_interval=args.sample_interval,
        relax_sampled_frames=CONFIG.relax_sampled_frames,
        sampled_frame_fmax=args.sampled_frame_fmax,
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
