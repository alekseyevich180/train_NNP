import csv
import os
from collections import Counter

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS
from ase.units import fs
from ase.neighborlist import neighbor_list

import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode


CONFIG = {
    "system": {
        "input_file": "acid.cif",
        "input_files": [
            "acid1.cif",
            "acid2.cif",
            "acid3.cif",
            "acid4.cif",
        ],
        "output_root": "acid_stability_scan",
        "substrate_symbols": ["Zn", "O"],
        "bottom_z_max": 9.0,
        "surface_relax_depth": 12.0,
        "fixed_z_lower_bound": 4.0,
        "fixed_z_upper_bound": 9.0,
    },
    "relaxation": {
        "surface_fmax": 0.05,
        "whole_fmax": 0.05,
        "snapshot_fmax": 0.05,
    },
    "temperature_scan": {
        "start": 280,
        "stop": 1080,
        "step": 50,
        "replicas": 1,
    },
    "md_control": {
        "equil_steps": 10000,
        "prod_steps": 20000,
        "timestep": 0.5,
        "tau_t": 100.0,
        "sample_interval": 1000,
    },
    "adsorption": {
        "enabled": True,
        "max_distance": 2.6,
    },
    "basin": {
        "energy_tolerance": 0.10,
    },
}


def build_calculator():
    estimator = Estimator(calc_mode=EstimatorCalcMode.PBE_U_PLUS_D3)
    return ASECalculator(estimator)


def temperature_grid():
    scan = CONFIG["temperature_scan"]
    return list(range(scan["start"], scan["stop"] + 1, scan["step"]))


def get_substrate_symbols():
    return set(CONFIG["system"]["substrate_symbols"])


def get_bottom_fixed_indices(atoms):
    z_max = CONFIG["system"]["bottom_z_max"]
    substrate_symbols = get_substrate_symbols()
    return [
        atom.index
        for atom in atoms
        if atom.symbol in substrate_symbols and atom.position[2] <= z_max
    ]


def get_aimd_fixed_indices(atoms):
    fixed_z_lower = CONFIG["system"]["fixed_z_lower_bound"]
    fixed_z_upper = CONFIG["system"]["fixed_z_upper_bound"]
    substrate_symbols = get_substrate_symbols()
    return [
        atom.index
        for atom in atoms
        if atom.symbol in substrate_symbols
        and fixed_z_lower <= atom.position[2] <= fixed_z_upper
    ]


def relax_surface(atoms):
    print("Starting surface relaxation...")

    fixed_z_lower = CONFIG["system"]["fixed_z_lower_bound"]
    fixed_z_upper = CONFIG["system"]["fixed_z_upper_bound"]
    surface_depth = CONFIG["system"]["surface_relax_depth"]
    substrate_symbols = get_substrate_symbols()

    max_z = np.max(atoms.positions[:, 2])
    surface_z = max_z - surface_depth

    freeze = []
    for atom in atoms:
        if atom.symbol not in substrate_symbols:
            freeze.append(atom.index)
        elif fixed_z_lower <= atom.position[2] <= fixed_z_upper:
            freeze.append(atom.index)
        elif atom.position[2] < surface_z:
            freeze.append(atom.index)

    atoms.set_constraint(FixAtoms(indices=freeze))

    opt = LBFGS(atoms, logfile="surface_relax.log")
    opt.run(fmax=CONFIG["relaxation"]["surface_fmax"])

    atoms.set_constraint()
    print("Surface relaxation finished")


def relax_whole_structure(atoms):
    print("Starting whole-structure relaxation...")

    fixed = get_bottom_fixed_indices(atoms)
    atoms.set_constraint(FixAtoms(indices=fixed))

    opt = LBFGS(atoms, logfile="whole_relax.log")
    opt.run(fmax=CONFIG["relaxation"]["whole_fmax"])

    atoms.set_constraint()
    print("Whole-structure relaxation finished")


def atom_symbol_indices(atoms, symbols):
    symbol_set = set(symbols)
    return [atom.index for atom in atoms if atom.symbol in symbol_set]


def is_adsorbed(atoms):
    if not CONFIG["adsorption"]["enabled"]:
        return True

    substrate_symbols = get_substrate_symbols()
    adsorbate_indices = [atom.index for atom in atoms if atom.symbol not in substrate_symbols]
    surface_indices = atom_symbol_indices(atoms, substrate_symbols)

    if not adsorbate_indices:
        return True
    if not surface_indices:
        return False

    cutoff = CONFIG["adsorption"]["max_distance"]
    cutoffs = np.full(len(atoms), cutoff / 2.0)
    i_list, j_list, distances = neighbor_list("ijd", atoms, cutoffs)

    for i_atom, j_atom, distance in zip(i_list, j_list, distances):
        if distance > cutoff:
            continue
        if i_atom in adsorbate_indices and j_atom in surface_indices:
            return True
        if j_atom in adsorbate_indices and i_atom in surface_indices:
            return True
    return False


def apply_constraints(atoms, fixed_indices):
    atoms.set_constraint(FixAtoms(indices=fixed_indices))


def assign_basin_id(relaxed_energy, basin_centers):
    tol = CONFIG["basin"]["energy_tolerance"]
    for basin_id, energy_center in enumerate(basin_centers):
        if abs(relaxed_energy - energy_center) <= tol:
            return basin_id
    basin_centers.append(relaxed_energy)
    return len(basin_centers) - 1


def quench_snapshot(sample_atoms, fixed_indices, out_dir, label):
    relaxed = sample_atoms.copy()
    relaxed.calc = build_calculator()
    apply_constraints(relaxed, fixed_indices)

    logfile = os.path.join(out_dir, f"{label}_relax.log")
    opt = LBFGS(relaxed, logfile=logfile)
    opt.run(fmax=CONFIG["relaxation"]["snapshot_fmax"])

    relaxed_energy = relaxed.get_potential_energy()
    relaxed_file = os.path.join(out_dir, f"{label}_relaxed.cif")
    write(relaxed_file, relaxed)

    return relaxed_energy, relaxed_file


def write_temperature_summary(summary_path, rows):
    fieldnames = [
        "temperature_K",
        "replica",
        "sample_id",
        "md_energy_eV",
        "relaxed_energy_eV",
        "adsorbed_md",
        "adsorbed_relaxed",
        "basin_id",
        "md_frame",
        "relaxed_frame",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single_temperature(base_atoms, temperature, replica_id, output_root):
    temp_dir = os.path.join(output_root, f"T_{temperature:04d}K", f"replica_{replica_id:02d}")
    md_dir = os.path.join(temp_dir, "md_frames")
    relaxed_dir = os.path.join(temp_dir, "relaxed_frames")
    os.makedirs(md_dir, exist_ok=True)
    os.makedirs(relaxed_dir, exist_ok=True)

    atoms = base_atoms.copy()
    atoms.calc = build_calculator()
    fixed_indices = get_bottom_fixed_indices(atoms)
    apply_constraints(atoms, fixed_indices)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = NVTBerendsen(
        atoms,
        timestep=CONFIG["md_control"]["timestep"] * fs,
        temperature_K=temperature,
        taut=CONFIG["md_control"]["tau_t"],
    )

    print(f"Equilibrating at {temperature} K, replica {replica_id}")
    dyn.run(CONFIG["md_control"]["equil_steps"])

    sample_interval = CONFIG["md_control"]["sample_interval"]
    prod_steps = CONFIG["md_control"]["prod_steps"]
    frame_rows = []
    basin_centers = []
    sample_counter = {"count": 0}

    def sample_frame():
        step = dyn.nsteps
        if step % sample_interval != 0:
            return

        sample_counter["count"] += 1
        sample_id = sample_counter["count"]
        label = f"T{temperature:04d}_R{replica_id:02d}_S{sample_id:03d}"

        md_atoms = atoms.copy()
        md_atoms.calc = build_calculator()
        apply_constraints(md_atoms, fixed_indices)

        md_energy = md_atoms.get_potential_energy()
        md_ads = is_adsorbed(md_atoms)
        md_file = os.path.join(md_dir, f"{label}_md.cif")
        write(md_file, md_atoms)

        relaxed_energy, relaxed_file = quench_snapshot(md_atoms, fixed_indices, relaxed_dir, label)
        relaxed_atoms = read(relaxed_file)
        relaxed_atoms.calc = build_calculator()
        relaxed_ads = is_adsorbed(relaxed_atoms)
        basin_id = assign_basin_id(relaxed_energy, basin_centers)

        frame_rows.append(
            {
                "temperature_K": temperature,
                "replica": replica_id,
                "sample_id": sample_id,
                "md_energy_eV": md_energy,
                "relaxed_energy_eV": relaxed_energy,
                "adsorbed_md": int(md_ads),
                "adsorbed_relaxed": int(relaxed_ads),
                "basin_id": basin_id,
                "md_frame": md_file,
                "relaxed_frame": relaxed_file,
            }
        )
        print(
            f"Sampled {label}: md_energy={md_energy:.6f} eV, "
            f"relaxed_energy={relaxed_energy:.6f} eV, basin={basin_id}"
        )

    dyn.attach(sample_frame, interval=1)

    print(f"Production run at {temperature} K, replica {replica_id}")
    dyn.run(prod_steps)

    summary_path = os.path.join(temp_dir, "samples_summary.csv")
    write_temperature_summary(summary_path, frame_rows)

    return frame_rows


def summarize_temperature(rows):
    if not rows:
        return None

    relaxed_energies = np.array([row["relaxed_energy_eV"] for row in rows], dtype=float)
    md_energies = np.array([row["md_energy_eV"] for row in rows], dtype=float)
    basin_counter = Counter(row["basin_id"] for row in rows)
    dominant_basin_id, dominant_count = basin_counter.most_common(1)[0]

    return {
        "temperature_K": rows[0]["temperature_K"],
        "num_samples": len(rows),
        "num_replicas": len(set(row["replica"] for row in rows)),
        "avg_md_energy_eV": float(md_energies.mean()),
        "min_md_energy_eV": float(md_energies.min()),
        "avg_relaxed_energy_eV": float(relaxed_energies.mean()),
        "lowest_relaxed_energy_eV": float(relaxed_energies.min()),
        "adsorbed_md_fraction": float(np.mean([row["adsorbed_md"] for row in rows])),
        "adsorbed_relaxed_fraction": float(np.mean([row["adsorbed_relaxed"] for row in rows])),
        "dominant_basin_id": dominant_basin_id,
        "dominant_basin_fraction": dominant_count / len(rows),
    }


def write_overall_summary(path, rows):
    fieldnames = [
        "temperature_K",
        "num_samples",
        "num_replicas",
        "avg_md_energy_eV",
        "min_md_energy_eV",
        "avg_relaxed_energy_eV",
        "lowest_relaxed_energy_eV",
        "adsorbed_md_fraction",
        "adsorbed_relaxed_fraction",
        "dominant_basin_id",
        "dominant_basin_fraction",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_input_files():
    system_cfg = CONFIG["system"]
    input_files = system_cfg.get("input_files") or []
    if input_files:
        return input_files
    return [system_cfg["input_file"]]


def structure_tag_from_path(input_file):
    return os.path.splitext(os.path.basename(input_file))[0]


def run_single_structure(input_file, batch_root):
    structure_tag = structure_tag_from_path(input_file)
    output_root = os.path.join(batch_root, structure_tag)
    os.makedirs(output_root, exist_ok=True)

    atoms = read(input_file)
    atoms.calc = build_calculator()
    atoms.pbc = True

    relax_surface(atoms)
    relax_whole_structure(atoms)

    prepared_path = os.path.join(output_root, "prepared_initial_structure.cif")
    write(prepared_path, atoms)

    overall_rows = []
    replicas = CONFIG["temperature_scan"]["replicas"]

    for temperature in temperature_grid():
        temperature_rows = []
        for replica_id in range(1, replicas + 1):
            rows = run_single_temperature(atoms, temperature, replica_id, output_root)
            temperature_rows.extend(rows)

        summary = summarize_temperature(temperature_rows)
        if summary is not None:
            overall_rows.append(summary)

    overall_summary_path = os.path.join(output_root, "overall_summary.csv")
    write_overall_summary(overall_summary_path, overall_rows)
    print(f"Finished {structure_tag}. Summary written to {overall_summary_path}")


def run():
    batch_root = CONFIG["system"]["output_root"]
    os.makedirs(batch_root, exist_ok=True)

    input_files = resolve_input_files()
    print(f"Batch start for {len(input_files)} structure(s)")

    for input_file in input_files:
        print(f"Processing structure: {input_file}")
        run_single_structure(input_file, batch_root)


if __name__ == "__main__":
    run()
