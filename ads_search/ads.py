import argparse
import io
from collections import Counter
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from ase import Atoms
from ase.filters import ExpCellFilter, StrainFilter
from ase.io import read, write
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True

CALCULATOR = None


def normalize_symbols(symbols: list[str]) -> list[str]:
    normalized = []
    for symbol in symbols:
        cleaned = symbol.strip().strip(",")
        if not cleaned:
            continue
        normalized.append(cleaned[0].upper() + cleaned[1:].lower())
    return normalized


def format_symbol_counts(atoms: Atoms) -> str:
    counts = Counter(atoms.get_chemical_symbols())
    return ", ".join(f"{symbol}:{counts[symbol]}" for symbol in sorted(counts))


def build_calculator(uma_model: str, device: str, include_d3: bool, checkpoint: Path | None = None):
    model_ref = str(checkpoint) if checkpoint is not None else uma_model
    calc = FAIRChemCalculator.from_model_checkpoint(model_ref, task_name="oc22", device=device)

    if include_d3:
        from ase.calculators.dftd3 import DFTD3

        calc = DFTD3(dft=calc)

    return calc


def get_opt_energy(atoms: Atoms, calculator, fmax: float = 1e-3, opt_mode: str = "normal") -> float:
    atoms.calc = calculator
    if opt_mode == "scale":
        optimizer = LBFGS(StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None)
    elif opt_mode == "all":
        optimizer = LBFGS(ExpCellFilter(atoms), logfile=None)
    else:
        optimizer = LBFGS(atoms, logfile=None)
    optimizer.run(fmax=fmax)
    return atoms.get_total_energy()


def atoms_to_json(atoms: Atoms) -> str:
    buffer = io.StringIO()
    write(buffer, atoms, format="json")
    return buffer.getvalue()


def json_to_atoms(atoms_str: str) -> Atoms:
    return read(io.StringIO(atoms_str), format="json")


def load_slab(surface_path: Path, calculator, fmax: float) -> tuple[Atoms, float]:
    slab = read(surface_path)
    energy = get_opt_energy(slab, calculator, fmax=fmax, opt_mode="normal")
    return slab, energy


def load_molecule(molecule_path: Path, calculator, fmax: float) -> tuple[Atoms, float]:
    mol = read(molecule_path)
    energy = get_opt_energy(mol, calculator, fmax=fmax)
    return mol, energy


def get_active_indices(
    atoms: Atoms, active_symbols: list[str], structure_path: Path | None = None
) -> list[int]:
    active_symbol_set = set(normalize_symbols(active_symbols))
    indices = [idx for idx, atom in enumerate(atoms) if atom.symbol in active_symbol_set]
    if not indices:
        available_symbols = sorted(set(atoms.get_chemical_symbols()))
        hint = ""
        if structure_path is not None and structure_path.suffix.lower() in {".vasp", ".poscar", ".contcar"}:
            hint = (
                " Hint: VASP/POSCAR files do not store element labels per atom. "
                "If you replaced Ni with Ru, you must also update the species/count header "
                "and regroup atoms by element so ASE reads Ru from the file."
            )
        raise ValueError(
            "No active-site atoms found for symbols: "
            f"{', '.join(active_symbol_set)}. "
            f"Available symbols in slab: {', '.join(available_symbols)}. "
            f"Symbol counts: {format_symbol_counts(atoms)}."
            f"{hint}"
        )
    return indices


def get_min_active_distance(combined: Atoms, n_slab_atoms: int, active_indices: list[int]) -> float:
    active_positions = combined.positions[active_indices]
    molecule_positions = combined.positions[n_slab_atoms:]
    deltas = active_positions[:, None, :] - molecule_positions[None, :, :]
    return float(np.linalg.norm(deltas, axis=2).min())


def objective(trial: optuna.Trial) -> float:
    slab = json_to_atoms(trial.study.user_attrs["slab"])
    e_slab = trial.study.user_attrs["E_slab"]
    mol = json_to_atoms(trial.study.user_attrs["mol"])
    e_mol = trial.study.user_attrs["E_mol"]
    output_dir = Path(trial.study.user_attrs["output_dir"])
    active_indices = trial.study.user_attrs["active_indices"]
    site_radius = float(trial.study.user_attrs["site_radius"])
    detach_cutoff = float(trial.study.user_attrs["detach_cutoff"])
    penalty_energy = float(trial.study.user_attrs["penalty_energy"])

    phi = 180.0 * trial.suggest_float("phi", -1.0, 1.0)
    theta = np.degrees(np.arccos(trial.suggest_float("theta", -1.0, 1.0)))
    psi = 180.0 * trial.suggest_float("psi", -1.0, 1.0)
    site_choice = trial.suggest_int("site_index", 0, len(active_indices) - 1)
    dx = trial.suggest_float("dx", -site_radius, site_radius)
    dy = trial.suggest_float("dy", -site_radius, site_radius)
    z_hig = trial.suggest_float("z_hig", 2.0, 6.0)

    mol.euler_rotate(phi=phi, theta=theta, psi=psi)

    active_index = active_indices[site_choice]
    active_position = slab.positions[active_index]
    mol.translate([active_position[0] + dx, active_position[1] + dy, 0.0])

    min_mol_z = np.min(mol.positions[:, 2])
    mol.translate([0.0, 0.0, active_position[2] + z_hig - min_mol_z])

    combined = slab + mol
    e_total = get_opt_energy(combined, CALCULATOR, fmax=1e-3)
    adsorption_energy = e_total - e_slab - e_mol
    structure_json = atoms_to_json(combined)
    write(output_dir / f"{trial.number}.cif", combined)
    trial.set_user_attr("structure", structure_json)

    min_active_distance = get_min_active_distance(combined, len(slab), active_indices)
    trial.set_user_attr("min_active_distance", min_active_distance)
    if min_active_distance > detach_cutoff:
        trial.set_user_attr("detached", True)
        return penalty_energy + min_active_distance
    trial.set_user_attr("detached", False)

    block_best_value = trial.study.user_attrs.get("block_best_value")
    if block_best_value is None or adsorption_energy < block_best_value:
        trial.study.set_user_attr("block_best_value", adsorption_energy)
        trial.study.set_user_attr("block_best_trial_number", trial.number)
        trial.study.set_user_attr("block_best_structure", structure_json)

    if (trial.number + 1) % 100 == 0:
        block_start = trial.number - 99
        block_end = trial.number
        block_best_structure = json_to_atoms(trial.study.user_attrs["block_best_structure"])
        phase_best_path = output_dir / f"trial_{block_start:03d}_{block_end:03d}_best.cif"
        write(phase_best_path, block_best_structure)
        trial.study.set_user_attr("block_best_value", None)
        trial.study.set_user_attr("block_best_trial_number", None)
        trial.study.set_user_attr("block_best_structure", None)

    return adsorption_energy


def save_trial_grid(study: optuna.Study, output_dir: Path) -> None:
    n_trials = len(study.trials)
    ncols = 10
    nrows = max(1, (n_trials + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 2 * nrows))
    axes = np.atleast_2d(axes)

    for idx, ax in enumerate(axes.flat):
        if idx >= n_trials:
            ax.set_axis_off()
            continue

        trial = study.trials[idx]
        structure = json_to_atoms(trial.user_attrs["structure"])
        cif_path = output_dir / f"{trial.number}.cif"
        png_path = output_dir / f"{trial.number}.png"
        write(cif_path, structure)
        write(png_path, structure, rotation="0x,0y,90z")
        ax.imshow(mpimg.imread(png_path))
        ax.set_axis_off()
        ax.set_title(str(trial.number))

    plt.tight_layout()
    plt.savefig(output_dir / "trial_grid.png")
    plt.close(fig)


def main():
    global CALCULATOR

    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, default=Path("surface-3O.cif"))
    parser.add_argument(
        "--molecules",
        type=Path,
        nargs="+",
        default=[Path("5-ketone.cif"), Path("2-6ketone.cif"), Path("3-6ketone.cif")],
    )
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--fmax", type=float, default=1e-4)
    parser.add_argument("--uma_model", type=str, default="uma-s-1p2")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--active_symbols", nargs="+", default=["Co"])
    parser.add_argument("--site_radius", type=float, default=2.0)
    parser.add_argument("--detach_cutoff", type=float, default=4.0)
    parser.add_argument("--penalty_energy", type=float, default=1e6)
    parser.add_argument("--include_d3", action="store_true")
    args = parser.parse_args()
    args.active_symbols = normalize_symbols(args.active_symbols)

    CALCULATOR = build_calculator(args.uma_model, args.device, args.include_d3, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for molecule_path in args.molecules:
        print(f"\nStarting optimization for {molecule_path.name}...\n")
        slab, e_slab = load_slab(args.surface, CALCULATOR, args.fmax)
        mol, e_mol = load_molecule(molecule_path, CALCULATOR, args.fmax)
        output_dir = args.output_dir / molecule_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Requested active symbols: {', '.join(args.active_symbols)}")
        print(f"Slab symbols: {format_symbol_counts(slab)}")
        active_indices = get_active_indices(slab, args.active_symbols, args.surface)

        study = optuna.create_study(direction="minimize")
        study.set_user_attr("slab", atoms_to_json(slab))
        study.set_user_attr("E_slab", e_slab)
        study.set_user_attr("mol", atoms_to_json(mol))
        study.set_user_attr("E_mol", e_mol)
        study.set_user_attr("output_dir", str(output_dir))
        study.set_user_attr("active_indices", active_indices)
        study.set_user_attr("site_radius", args.site_radius)
        study.set_user_attr("detach_cutoff", args.detach_cutoff)
        study.set_user_attr("penalty_energy", args.penalty_energy)
        study.optimize(objective, n_trials=args.n_trials)

        print(f"Best trial for {molecule_path.name}: #{study.best_trial.number}")
        print(f"Adsorption energy: {study.best_value:.6f} eV")

        optuna.visualization.plot_optimization_history(study).write_html(
            output_dir / "optimization_history.html"
        )
        optuna.visualization.plot_slice(study).write_html(output_dir / "optimization_slice.html")

        best_structure = json_to_atoms(study.best_trial.user_attrs["structure"])
        write(output_dir / "best_trial.cif", best_structure)

        trial_data = []
        for trial in study.trials:
            trial_data.append({"trial": trial.number, "energy": trial.value, **trial.params})
        pd.DataFrame(trial_data).to_csv(output_dir / "data.csv", index=False)

        save_trial_grid(study, output_dir)


if __name__ == "__main__":
    main()
