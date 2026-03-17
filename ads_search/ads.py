import argparse
import io
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


def build_calculator(uma_model: str, device: str, include_d3: bool, checkpoint: Path | None = None):
    model_ref = str(checkpoint) if checkpoint is not None else uma_model
    calc = FAIRChemCalculator.from_model_checkpoint(model_ref, task_name="omat", device=device)

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


def objective(trial: optuna.Trial) -> float:
    slab = json_to_atoms(trial.study.user_attrs["slab"])
    e_slab = trial.study.user_attrs["E_slab"]
    mol = json_to_atoms(trial.study.user_attrs["mol"])
    e_mol = trial.study.user_attrs["E_mol"]

    phi = 180.0 * trial.suggest_float("phi", -1.0, 1.0)
    theta = np.degrees(np.arccos(trial.suggest_float("theta", -1.0, 1.0)))
    psi = 180.0 * trial.suggest_float("psi", -1.0, 1.0)
    x_pos = trial.suggest_float("x_pos", 0.0, 0.5)
    y_pos = trial.suggest_float("y_pos", 0.0, 0.5)
    z_hig = trial.suggest_float("z_hig", 2.0, 6.0)

    mol.euler_rotate(phi=phi, theta=theta, psi=psi)

    xy_position = np.matmul([x_pos, y_pos, 0.0], slab.cell)[:2]
    mol.translate([xy_position[0], xy_position[1], 0.0])

    max_slab_z = np.max(slab.positions[:, 2])
    min_mol_z = np.min(mol.positions[:, 2])
    mol.translate([0.0, 0.0, max_slab_z + z_hig - min_mol_z])

    combined = slab + mol
    e_total = get_opt_energy(combined, CALCULATOR, fmax=1e-3)
    trial.set_user_attr("structure", atoms_to_json(combined))
    return e_total - e_slab - e_mol


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
    parser.add_argument("--uma_model", type=str, default="uma-s-1p1")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--include_d3", action="store_true")
    args = parser.parse_args()

    CALCULATOR = build_calculator(args.uma_model, args.device, args.include_d3, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for molecule_path in args.molecules:
        print(f"\nStarting optimization for {molecule_path.name}...\n")
        slab, e_slab = load_slab(args.surface, CALCULATOR, args.fmax)
        mol, e_mol = load_molecule(molecule_path, CALCULATOR, args.fmax)

        study = optuna.create_study(direction="minimize")
        study.set_user_attr("slab", atoms_to_json(slab))
        study.set_user_attr("E_slab", e_slab)
        study.set_user_attr("mol", atoms_to_json(mol))
        study.set_user_attr("E_mol", e_mol)
        study.optimize(objective, n_trials=args.n_trials)

        print(f"Best trial for {molecule_path.name}: #{study.best_trial.number}")
        print(f"Adsorption energy: {study.best_value:.6f} eV")

        output_dir = args.output_dir / molecule_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

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
