import io
import math
import random
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import LBFGS
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode


# Copy this script into Matlantis and edit this block for your calculation.
CONFIG = {
    "input": {
        "structure": "ZnO.cif",
        "output_dir": "output",
        "structure_prefix": "ov",
    },
    "calculator": {
        # PBE is the current mode used with PFP v8 in Matlantis examples.
        "calc_mode": EstimatorCalcMode.PBE,
        "model_version": "v8.0.0",
    },
    "vacancy_search": {
        # A separate Optuna study is run for each vacancy count.
        "vacancy_counts": [1, 2, 3, 4, 5],
        "n_trials_per_count": 100,
        # Only O atoms in this fractional-z interval can be removed.
        "z_frac_range": [0.0, 1.0],
        "seed": 7,
    },
    "energy": {
        # "o2": mu_O = 1/2 E(O2) + delta_mu_o (recommended O-rich reference).
        # "manual": use manual_mu_o directly in the calculator's energy scale.
        "mu_o_mode": "o2",
        "manual_mu_o": 0.0,
        "delta_mu_o": 0.0,
        "o2_bond_length": 1.21,
        "o2_vacuum": 8.0,
    },
    "relaxation": {
        # Force convergence threshold in eV/Angstrom.
        "fmax": 0.05,
        "max_steps": 1000,
        # Non-converged trials are recorded but cannot be selected as best when
        # at least one converged trial is available.
        "require_clean_convergence": True,
    },
    "constraints": {
        # For a slab, set a fractional z value to fix atoms below it, e.g. 0.2.
        # Keep None for bulk ZnO or when every atom should be relaxed.
        "fix_z_frac_below": None,
    },
    "output": {
        "write_all_structures": False,
    },
}


CALCULATOR = None


def build_calculator(calc_mode: EstimatorCalcMode, model_version: str) -> ASECalculator:
    estimator = Estimator(calc_mode=calc_mode, model_version=model_version)
    return ASECalculator(estimator)


def optimize_energy(
    atoms: Atoms,
    calculator: ASECalculator,
    fmax: float,
    max_steps: int,
) -> tuple[float, bool, int]:
    atoms.calc = calculator
    optimizer = LBFGS(atoms, logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
    energy = float(atoms.get_potential_energy())
    return energy, converged, int(optimizer.get_number_of_steps())


def selectable_oxygen_indices(
    atoms: Atoms,
    z_frac_min: float,
    z_frac_max: float,
) -> list[int]:
    scaled_positions = atoms.get_scaled_positions(wrap=True)
    indices = [
        atom.index
        for atom in atoms
        if atom.symbol == "O"
        and z_frac_min <= scaled_positions[atom.index, 2] <= z_frac_max
    ]
    if not indices:
        raise ValueError(
            "No oxygen atoms were found in the requested fractional-z range "
            f"[{z_frac_min}, {z_frac_max}]."
        )
    return indices


def combination_from_rank(
    items: Sequence[int],
    count: int,
    rank: int,
) -> tuple[int, ...]:
    """Return one lexicographically ranked combination without listing all combinations."""
    item_count = len(items)
    if count < 1 or count > item_count:
        raise ValueError("Invalid combination size.")
    total = math.comb(item_count, count)
    if rank < 0 or rank >= total:
        raise ValueError(f"Combination rank must be in [0, {total - 1}].")

    selected: list[int] = []
    start = 0
    remaining = count
    current_rank = rank

    while remaining:
        for position in range(start, item_count - remaining + 1):
            combinations_after = math.comb(item_count - position - 1, remaining - 1)
            if current_rank < combinations_after:
                selected.append(int(items[position]))
                start = position + 1
                remaining -= 1
                break
            current_rank -= combinations_after

    return tuple(selected)


def remove_oxygen_atoms(atoms: Atoms, vacancy_indices: Sequence[int]) -> Atoms:
    defect = atoms.copy()
    for index in sorted(vacancy_indices, reverse=True):
        del defect[index]
    return defect


def apply_bottom_constraint(atoms: Atoms, z_frac_below: float | None) -> int:
    if z_frac_below is None:
        return 0
    if not 0.0 <= z_frac_below <= 1.0:
        raise ValueError("fix_z_frac_below must be None or a value in [0, 1].")
    scaled_z = atoms.get_scaled_positions(wrap=True)[:, 2]
    fixed_indices = [index for index, z_value in enumerate(scaled_z) if z_value <= z_frac_below]
    if fixed_indices:
        atoms.set_constraint(FixAtoms(indices=fixed_indices))
    return len(fixed_indices)


def unique_pattern_ids(total_patterns: int, sample_count: int, seed: int) -> list[int]:
    """Sample unique integers without constructing the complete integer range."""
    if sample_count >= total_patterns:
        return list(range(total_patterns))

    rng = random.Random(seed)
    selected: set[int] = set()
    # Floyd's algorithm: O(sample_count) memory even when total_patterns is large.
    for value in range(total_patterns - sample_count, total_patterns):
        candidate = rng.randrange(value + 1)
        selected.add(value if candidate in selected else candidate)
    return sorted(selected)


def determine_oxygen_chemical_potential(
    calculator: ASECalculator,
    output_root: Path,
    fmax: float,
    max_steps: int,
) -> tuple[float, dict[str, float | str | bool | int]]:
    energy_config = CONFIG["energy"]
    mode = str(energy_config["mu_o_mode"]).strip().lower()

    if mode == "manual":
        mu_o = float(energy_config["manual_mu_o"])
        return mu_o, {"mu_o_mode": mode, "mu_O_eV": mu_o}
    if mode != "o2":
        raise ValueError("mu_o_mode must be either 'o2' or 'manual'.")

    bond_length = float(energy_config["o2_bond_length"])
    vacuum = float(energy_config["o2_vacuum"])
    delta_mu_o = float(energy_config["delta_mu_o"])
    if bond_length <= 0.0 or vacuum <= 0.0:
        raise ValueError("o2_bond_length and o2_vacuum must be positive.")
    if delta_mu_o > 0.0:
        raise ValueError("delta_mu_o must be <= 0 eV relative to the O-rich limit.")

    oxygen = Atoms(
        "O2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]],
        pbc=False,
    )
    oxygen.center(vacuum=vacuum)
    oxygen_energy, converged, steps = optimize_energy(
        oxygen,
        calculator,
        fmax,
        max_steps,
    )
    write(output_root / "o2_reference.xyz", oxygen)
    if not converged:
        raise RuntimeError(
            "The O2 reference did not converge. Increase max_steps or inspect the model settings."
        )

    mu_o = 0.5 * oxygen_energy + delta_mu_o
    return mu_o, {
        "mu_o_mode": mode,
        "E_O2_eV": oxygen_energy,
        "delta_mu_O_eV": delta_mu_o,
        "mu_O_eV": mu_o,
        "o2_converged": converged,
        "o2_optimization_steps": steps,
    }


def vacancy_position_features(
    reference: Atoms,
    vacancy_indices: Sequence[int],
) -> dict[str, float]:
    selection = list(vacancy_indices)
    scaled = reference.get_scaled_positions(wrap=True)[selection]
    cartesian = reference.positions[selection]
    mean_scaled = scaled.mean(axis=0)
    mean_cartesian = cartesian.mean(axis=0)
    return {
        "vacancy_x_frac": float(mean_scaled[0]),
        "vacancy_y_frac": float(mean_scaled[1]),
        "vacancy_z_frac": float(mean_scaled[2]),
        "vacancy_x_ang": float(mean_cartesian[0]),
        "vacancy_y_ang": float(mean_cartesian[1]),
        "vacancy_z_ang": float(mean_cartesian[2]),
    }


def atoms_to_json(atoms: Atoms) -> str:
    buffer = io.StringIO()
    write(buffer, atoms, format="json")
    return buffer.getvalue()


def json_to_atoms(text: str) -> Atoms:
    return read(io.StringIO(text), format="json")


def trial_table(study: optuna.Study) -> pd.DataFrame:
    records = []
    for trial in study.trials:
        if trial.value is None:
            continue
        records.append(
            {
                "trial": trial.number,
                "pattern_id": trial.params.get("pattern_id"),
                "vacancy_count": trial.user_attrs.get("vacancy_count"),
                "vacancy_indices": trial.user_attrs.get("vacancy_indices"),
                "vacancy_concentration": trial.user_attrs.get("vacancy_concentration"),
                "E_clean_eV": trial.user_attrs.get("E_clean_eV"),
                "E_defect_eV": trial.user_attrs.get("E_defect_eV"),
                "mu_O_eV": trial.user_attrs.get("mu_O_eV"),
                "E_vac_eV": trial.value,
                "E_vac_per_vacancy_eV": trial.user_attrs.get(
                    "E_vac_per_vacancy_eV"
                ),
                "converged": trial.user_attrs.get("converged"),
                "optimization_steps": trial.user_attrs.get("optimization_steps"),
                "reused_result": trial.user_attrs.get("reused_result", False),
                "structure_path": trial.user_attrs.get("structure_path", ""),
                "vacancy_x_frac": trial.user_attrs.get("vacancy_x_frac"),
                "vacancy_y_frac": trial.user_attrs.get("vacancy_y_frac"),
                "vacancy_z_frac": trial.user_attrs.get("vacancy_z_frac"),
                "vacancy_x_ang": trial.user_attrs.get("vacancy_x_ang"),
                "vacancy_y_ang": trial.user_attrs.get("vacancy_y_ang"),
                "vacancy_z_ang": trial.user_attrs.get("vacancy_z_ang"),
            }
        )
    return pd.DataFrame(records).sort_values("E_vac_eV").reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].plot(
        summary["vacancy_count"],
        summary["best_E_vac_eV"],
        marker="o",
    )
    axes[0].set_xlabel("Number of oxygen vacancies")
    axes[0].set_ylabel("Best vacancy formation energy (eV)")
    axes[0].set_title("Best energy for each composition")

    axes[1].plot(
        summary["vacancy_count"],
        summary["best_E_vac_per_vacancy_eV"],
        marker="o",
    )
    axes[1].set_xlabel("Number of oxygen vacancies")
    axes[1].set_ylabel("Formation energy per vacancy (eV)")
    axes[1].set_title("Energy per oxygen vacancy")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    global CALCULATOR

    input_path = Path(CONFIG["input"]["structure"])
    output_root = Path(CONFIG["input"]["output_dir"])
    structure_prefix = str(CONFIG["input"]["structure_prefix"])
    calc_mode = CONFIG["calculator"]["calc_mode"]
    model_version = str(CONFIG["calculator"]["model_version"])
    vacancy_counts = sorted(
        {int(value) for value in CONFIG["vacancy_search"]["vacancy_counts"]}
    )
    n_trials_per_count = int(CONFIG["vacancy_search"]["n_trials_per_count"])
    z_frac_min, z_frac_max = [
        float(value) for value in CONFIG["vacancy_search"]["z_frac_range"]
    ]
    seed = int(CONFIG["vacancy_search"]["seed"])
    fmax = float(CONFIG["relaxation"]["fmax"])
    max_steps = int(CONFIG["relaxation"]["max_steps"])
    require_clean_convergence = bool(
        CONFIG["relaxation"]["require_clean_convergence"]
    )
    fix_z_frac_below = CONFIG["constraints"]["fix_z_frac_below"]
    if fix_z_frac_below is not None:
        fix_z_frac_below = float(fix_z_frac_below)
    write_all = bool(CONFIG["output"]["write_all_structures"])

    if not input_path.exists():
        raise FileNotFoundError(f"Input structure not found: {input_path}")
    if not 0.0 <= z_frac_min <= z_frac_max <= 1.0:
        raise ValueError("z_frac_range must satisfy 0 <= min <= max <= 1.")
    if not vacancy_counts or min(vacancy_counts) < 1:
        raise ValueError("vacancy_counts must contain positive integers.")
    if n_trials_per_count < 1:
        raise ValueError("n_trials_per_count must be at least 1.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    output_root.mkdir(parents=True, exist_ok=True)
    CALCULATOR = build_calculator(calc_mode, model_version)

    clean = read(input_path)
    all_oxygen_count = sum(symbol == "O" for symbol in clean.get_chemical_symbols())
    if all_oxygen_count == 0:
        raise ValueError("The input structure contains no oxygen atoms.")
    oxygen_indices = selectable_oxygen_indices(clean, z_frac_min, z_frac_max)
    if max(vacancy_counts) > len(oxygen_indices):
        raise ValueError(
            "The largest vacancy count exceeds the number of selectable oxygen atoms "
            f"({len(oxygen_indices)})."
        )

    print(f"Input structure: {input_path}")
    print(f"Output directory: {output_root}")
    print(f"Matlantis calc_mode: {calc_mode}")
    print(f"Matlantis model_version: {model_version}")
    print(f"Total O atoms: {all_oxygen_count}")
    print(f"Selectable O atoms: {len(oxygen_indices)}")

    clean_relaxed = clean.copy()
    fixed_atom_count = apply_bottom_constraint(clean_relaxed, fix_z_frac_below)
    print(f"Fixed atoms: {fixed_atom_count}")
    clean_energy, clean_converged, clean_steps = optimize_energy(
        clean_relaxed,
        CALCULATOR,
        fmax,
        max_steps,
    )
    write(output_root / "clean_relaxed.cif", clean_relaxed)
    print(
        f"Clean energy: {clean_energy:.8f} eV; "
        f"converged={clean_converged}; steps={clean_steps}"
    )
    if not clean_converged:
        message = "The clean structure did not reach the requested fmax."
        if require_clean_convergence:
            raise RuntimeError(message)
        print(f"WARNING: {message}")

    mu_o, mu_o_metadata = determine_oxygen_chemical_potential(
        CALCULATOR,
        output_root,
        fmax,
        max_steps,
    )
    pd.DataFrame([mu_o_metadata]).to_csv(
        output_root / "oxygen_chemical_potential.csv",
        index=False,
    )
    print(f"Oxygen chemical potential: {mu_o:.8f} eV ({mu_o_metadata['mu_o_mode']})")

    summary_records = []

    for vacancy_count in vacancy_counts:
        count_dir = output_root / f"vacancy_count_{vacancy_count}"
        structures_dir = count_dir / "structures"
        count_dir.mkdir(parents=True, exist_ok=True)
        if write_all:
            structures_dir.mkdir(parents=True, exist_ok=True)

        total_patterns = math.comb(len(oxygen_indices), vacancy_count)
        trial_count = min(n_trials_per_count, total_patterns)
        print(
            f"\nVacancy count {vacancy_count}: {total_patterns} possible patterns; "
            f"running {trial_count} Optuna trials."
        )

        def objective(trial: optuna.Trial) -> float:
            pattern_id = trial.suggest_int("pattern_id", 0, total_patterns - 1)

            vacancy_indices = combination_from_rank(
                oxygen_indices,
                vacancy_count,
                pattern_id,
            )
            defect = remove_oxygen_atoms(clean_relaxed, vacancy_indices)
            defect_energy, converged, steps = optimize_energy(
                defect,
                CALCULATOR,
                fmax,
                max_steps,
            )
            vacancy_energy = defect_energy - clean_energy + vacancy_count * mu_o
            vacancy_concentration = vacancy_count / all_oxygen_count

            structure_path = structures_dir / (
                f"{structure_prefix}_m{vacancy_count}_trial_{trial.number:04d}.cif"
            )
            if write_all:
                write(structure_path, defect)

            attrs = {
                "vacancy_count": vacancy_count,
                "vacancy_indices": " ".join(map(str, vacancy_indices)),
                "vacancy_concentration": vacancy_concentration,
                "E_clean_eV": clean_energy,
                "E_defect_eV": defect_energy,
                "mu_O_eV": mu_o,
                "E_vac_per_vacancy_eV": vacancy_energy / vacancy_count,
                "converged": converged,
                "optimization_steps": steps,
                "structure_path": str(structure_path) if write_all else "",
                "structure_json": atoms_to_json(defect),
                **vacancy_position_features(clean_relaxed, vacancy_indices),
            }
            for key, value in attrs.items():
                trial.set_user_attr(key, value)

            print(
                f"trial={trial.number:04d} pattern={pattern_id} "
                f"indices={attrs['vacancy_indices']} E_vac={vacancy_energy:.8f} eV "
                f"converged={converged}"
            )
            return vacancy_energy

        pattern_ids = unique_pattern_ids(
            total_patterns,
            trial_count,
            seed + vacancy_count,
        )
        # A grid over a unique random subset prevents duplicated expensive relaxations.
        sampler = optuna.samplers.GridSampler(
            {"pattern_id": pattern_ids},
            seed=seed + vacancy_count,
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=f"oxygen_vacancy_m{vacancy_count}",
        )
        study.optimize(objective, n_trials=trial_count)

        table = trial_table(study)
        table.to_csv(count_dir / "trials.csv", index=False)
        optuna.visualization.plot_optimization_history(study).write_html(
            count_dir / "optimization_history.html"
        )

        converged_trials = [
            trial
            for trial in study.trials
            if trial.value is not None and trial.user_attrs.get("converged") is True
        ]
        if not converged_trials:
            raise RuntimeError(
                f"No converged structures were found for vacancy_count={vacancy_count}. "
                "Increase max_steps or loosen fmax."
            )
        best_trial = min(converged_trials, key=lambda trial: float(trial.value))
        best_structure = json_to_atoms(best_trial.user_attrs["structure_json"])
        best_path = count_dir / f"best_vacancy_count_{vacancy_count}.cif"
        write(best_path, best_structure)

        summary_records.append(
            {
                "vacancy_count": vacancy_count,
                "vacancy_concentration": vacancy_count / all_oxygen_count,
                "possible_patterns": total_patterns,
                "completed_trials": len(table),
                "best_trial": best_trial.number,
                "best_pattern_id": best_trial.params["pattern_id"],
                "best_vacancy_indices": best_trial.user_attrs["vacancy_indices"],
                "best_E_vac_eV": best_trial.value,
                "best_E_vac_per_vacancy_eV": best_trial.value / vacancy_count,
                "best_converged": best_trial.user_attrs["converged"],
                "best_structure_path": str(best_path),
            }
        )
        print(
            f"Best for m={vacancy_count}: indices="
            f"{best_trial.user_attrs['vacancy_indices']}; "
            f"E_vac={best_trial.value:.8f} eV"
        )

    summary = pd.DataFrame(summary_records).sort_values("vacancy_count")
    summary_path = output_root / "vacancy_search_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, output_root / "vacancy_search_summary.png")

    print("\nFinished Optuna oxygen-vacancy search.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
