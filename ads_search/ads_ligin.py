import io
from collections import Counter
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from ase import Atoms
from ase.filters import ExpCellFilter, StrainFilter
from ase.io import read, write
from ase.optimize import LBFGS
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode


CONFIG = {
    "input": {
        "surface": "surface-3O.cif",
        "molecules": ["5-ketone.cif", "2-6ketone.cif", "3-6ketone.cif"],
        "output_dir": "output_matlantis",
    },
    "calculator": {
        "calc_mode": EstimatorCalcMode.CRYSTAL_U0,
        "model_version": "v3.0.0",
    },
    "optimization": {
        "n_trials": 300,
        "fmax": 1e-4,
        "max_steps": 200,
    },
    "adsorption": {
        "z_height_range": [2.0, 6.0],
    },
}


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


def build_calculator(calc_mode: EstimatorCalcMode, model_version: str) -> ASECalculator:
    estimator = Estimator(calc_mode=calc_mode, model_version=model_version)
    return ASECalculator(estimator)


def get_opt_energy(
    atoms: Atoms,
    calculator,
    fmax: float = 1e-3,
    opt_mode: str = "normal",
    max_steps: int | None = None,
) -> float:
    atoms.calc = calculator
    if opt_mode == "scale":
        optimizer = LBFGS(StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None)
    elif opt_mode == "all":
        optimizer = LBFGS(ExpCellFilter(atoms), logfile=None)
    else:
        optimizer = LBFGS(atoms, logfile=None)
    optimizer.run(fmax=fmax, steps=max_steps)
    return atoms.get_total_energy()


def atoms_to_json(atoms: Atoms) -> str:
    buffer = io.StringIO()
    write(buffer, atoms, format="json")
    return buffer.getvalue()


def json_to_atoms(atoms_str: str) -> Atoms:
    return read(io.StringIO(atoms_str), format="json")


def load_slab(surface_path: Path, calculator, fmax: float, max_steps: int | None) -> tuple[Atoms, float]:
    slab = read(surface_path)
    energy = get_opt_energy(slab, calculator, fmax=fmax, opt_mode="normal", max_steps=max_steps)
    return slab, energy


def load_molecule(
    molecule_path: Path, calculator, fmax: float, max_steps: int | None
) -> tuple[Atoms, float]:
    mol = read(molecule_path)
    energy = get_opt_energy(mol, calculator, fmax=fmax, max_steps=max_steps)
    return mol, energy


def objective(trial: optuna.Trial) -> float:
    slab = json_to_atoms(trial.study.user_attrs["slab"])
    e_slab = trial.study.user_attrs["E_slab"]
    mol = json_to_atoms(trial.study.user_attrs["mol"])
    e_mol = trial.study.user_attrs["E_mol"]
    output_dir = Path(trial.study.user_attrs["output_dir"])
    max_steps = trial.study.user_attrs["max_steps"]
    z_min = float(trial.study.user_attrs["z_height_min"])
    z_max = float(trial.study.user_attrs["z_height_max"])

    phi = 180.0 * trial.suggest_float("phi", -1.0, 1.0)
    theta = np.degrees(np.arccos(trial.suggest_float("theta", -1.0, 1.0)))
    psi = 180.0 * trial.suggest_float("psi", -1.0, 1.0)
    x_frac = trial.suggest_float("x_frac", 0.0, 1.0)
    y_frac = trial.suggest_float("y_frac", 0.0, 1.0)
    z_hig = trial.suggest_float("z_hig", z_min, z_max)

    mol.euler_rotate(phi=phi, theta=theta, psi=psi)

    xy_position = np.matmul([x_frac, y_frac, 0.0], slab.cell)[:2]
    mol.translate([xy_position[0], xy_position[1], 0.0])

    max_slab_z = np.max(slab.positions[:, 2])
    min_mol_z = np.min(mol.positions[:, 2])
    mol.translate([0.0, 0.0, max_slab_z + z_hig - min_mol_z])

    combined = slab + mol
    e_total = get_opt_energy(combined, CALCULATOR, fmax=1e-3, max_steps=max_steps)
    adsorption_energy = e_total - e_slab - e_mol
    structure_json = atoms_to_json(combined)
    write(output_dir / f"{trial.number}.cif", combined)
    trial.set_user_attr("structure", structure_json)

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


def main() -> None:
    global CALCULATOR

    surface = Path(CONFIG["input"]["surface"])
    molecules = [Path(path) for path in CONFIG["input"]["molecules"]]
    output_root = Path(CONFIG["input"]["output_dir"])
    calc_mode = CONFIG["calculator"]["calc_mode"]
    model_version = str(CONFIG["calculator"]["model_version"])
    n_trials = int(CONFIG["optimization"]["n_trials"])
    fmax = float(CONFIG["optimization"]["fmax"])
    max_steps = int(CONFIG["optimization"]["max_steps"])
    z_height_min, z_height_max = CONFIG["adsorption"]["z_height_range"]

    CALCULATOR = build_calculator(calc_mode, model_version)
    output_root.mkdir(parents=True, exist_ok=True)

    for molecule_path in molecules:
        print(f"\nStarting optimization for {molecule_path.name}...\n")
        slab, e_slab = load_slab(surface, CALCULATOR, fmax, max_steps)
        mol, e_mol = load_molecule(molecule_path, CALCULATOR, fmax, max_steps)
        output_dir = output_root / molecule_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Matlantis calc_mode: {calc_mode}")
        print(f"Matlantis model_version: {model_version}")
        print(f"Slab symbols: {format_symbol_counts(slab)}")
        print(f"Optimization step limit per structure: {max_steps}")
        print("Sampling mode: molecule is translated over the full surface cell using x_frac/y_frac in [0, 1].")

        study = optuna.create_study(direction="minimize")
        study.set_user_attr("slab", atoms_to_json(slab))
        study.set_user_attr("E_slab", e_slab)
        study.set_user_attr("mol", atoms_to_json(mol))
        study.set_user_attr("E_mol", e_mol)
        study.set_user_attr("output_dir", str(output_dir))
        study.set_user_attr("max_steps", max_steps)
        study.set_user_attr("z_height_min", float(z_height_min))
        study.set_user_attr("z_height_max", float(z_height_max))
        study.optimize(objective, n_trials=n_trials)

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
