from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs

from share.pfp import build_pfp_calculator
from share.tdbb import (
    BiasedCalculator,
    DeepMDWriter,
    TDBBBias,
    TDBBParameters,
    current_pair_distances,
    get_aimd_fixed_indices,
    get_reactive_pairs,
    get_template_target_distances,
    get_template_target_key,
    load_restart,
    load_template_data,
    make_md_segments,
    relax_surface,
    relax_whole_structure,
    resolve_existing_path,
    resolve_output_path,
    save_restart,
)


CONFIG = {
    "system": {
        "input_file": "ketone.cif",
        "fixed_z_lower_bound": 4.0,
        "fixed_z_upper_bound": 9.0,
        "surface_relax_depth": 12.0,
        "output_root": "acid_AIMD_dataset_test",
        "restart_from": None,
        "pbc": True,
        "pair_mode": "template",
        "reactive_pairs": "0-10",
        "reactive_symbols": "C-O",
        "pair_cutoff_ang": 4.0,
        "template_dir": "interm",
        "template_bond_symbols": "C-O,C=C,C-C",
        "template_bond_cutoff_ang": 2.2,
        "template_c_c_double_range_ang": (1.15, 1.45),
        "template_c_c_single_range_ang": (1.45, 1.70),
        "functional_o_c_cutoff_ang": 1.75,
        "functional_c_c_shell_cutoff_ang": 1.75,
        "functional_c_c_pair_cutoff_ang": 4.0,
        "template_pair_source": "functional-cc-and-co",
        "template_target_mode": "min",
        "double_bond_c_c_range_ang": (1.15, 1.45),
        "existing_c_o_bond_cutoff_ang": 1.65,
        "o_h_bond_cutoff_ang": 1.20,
        "enol_c_o_bond_cutoff_ang": 1.65,
    },
    "relaxation": {
        "surface_fmax": 0.05,
        "whole_fmax": 0.05,
    },
    "md_control": {
        "initial_temp": 280,
        "final_temp": 1080,
        "ramp_interval": 100,
        "ramp_steps": 20000,
        "stab_steps": 10000,
        "prod_steps": 4000000,
        "timestep": 0.5,
        "tau_t": 100.0,
    },
    "pfp": {
        "calc_mode": "PBE_U_PLUS_D3",
    },
    "tdbb": {
        "gamma": 1.0,
        "f1_max": 250.0,
        "f2": 10.0,
        "target_scale": 0.60,
        "default_target": 1.5,
    },
    "output": {
        "save_interval": 100,
        "deepmd_dir": "deepmd_dataset",
        "cif_dir": "cif_frames",
        "restart_dir": "restart_checkpoints",
        "deepmd_set_interval": 100000,
        "cif_set_interval": 100000,
        "restart_interval": 200000,
        "template_progress": "template_progress.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learning/test entry point for the TDBB AIMD workflow using shared helpers."
    )
    parser.add_argument("-i", "--input", default=CONFIG["system"]["input_file"], help="Initial structure file.")
    parser.add_argument("--output-root", default=CONFIG["system"]["output_root"], help="Root directory for outputs.")
    parser.add_argument("--restart-from", default=CONFIG["system"]["restart_from"], help="Checkpoint directory or latest_checkpoint.txt.")
    parser.add_argument("--template-progress", default=CONFIG["output"]["template_progress"], help="Template progress CSV name.")
    parser.add_argument("--pair-mode", choices=["manual", "symbols", "double-bond-co", "template"], default=CONFIG["system"]["pair_mode"])
    parser.add_argument("--pairs", default=CONFIG["system"]["reactive_pairs"])
    parser.add_argument("--symbols", default=CONFIG["system"]["reactive_symbols"])
    parser.add_argument("--pair-cutoff", type=float, default=CONFIG["system"]["pair_cutoff_ang"])
    parser.add_argument("--template-dir", default=CONFIG["system"]["template_dir"])
    parser.add_argument("--template-bond-symbols", default=CONFIG["system"]["template_bond_symbols"])
    parser.add_argument("--template-bond-cutoff", type=float, default=CONFIG["system"]["template_bond_cutoff_ang"])
    parser.add_argument("--template-double-min", type=float, default=CONFIG["system"]["template_c_c_double_range_ang"][0])
    parser.add_argument("--template-double-max", type=float, default=CONFIG["system"]["template_c_c_double_range_ang"][1])
    parser.add_argument("--template-single-min", type=float, default=CONFIG["system"]["template_c_c_single_range_ang"][0])
    parser.add_argument("--template-single-max", type=float, default=CONFIG["system"]["template_c_c_single_range_ang"][1])
    parser.add_argument(
        "--template-pair-source",
        choices=["manual", "symbols", "double-bond-co", "double-bond-and-co", "functional-cc", "functional-cc-and-co"],
        default=CONFIG["system"]["template_pair_source"],
    )
    parser.add_argument("--template-target-mode", choices=["min", "mean"], default=CONFIG["system"]["template_target_mode"])
    parser.add_argument("--double-bond-min", type=float, default=CONFIG["system"]["double_bond_c_c_range_ang"][0])
    parser.add_argument("--double-bond-max", type=float, default=CONFIG["system"]["double_bond_c_c_range_ang"][1])
    parser.add_argument("--existing-c-o-cutoff", type=float, default=CONFIG["system"]["existing_c_o_bond_cutoff_ang"])
    parser.add_argument("--functional-o-c-cutoff", type=float, default=CONFIG["system"]["functional_o_c_cutoff_ang"])
    parser.add_argument("--functional-c-c-shell-cutoff", type=float, default=CONFIG["system"]["functional_c_c_shell_cutoff_ang"])
    parser.add_argument("--functional-c-c-pair-cutoff", type=float, default=CONFIG["system"]["functional_c_c_pair_cutoff_ang"])
    parser.add_argument("--o-h-cutoff", type=float, default=CONFIG["system"]["o_h_bond_cutoff_ang"])
    parser.add_argument("--enol-c-o-cutoff", type=float, default=CONFIG["system"]["enol_c_o_bond_cutoff_ang"])
    parser.add_argument("--gamma", type=float, default=CONFIG["tdbb"]["gamma"])
    parser.add_argument("--f1-max", type=float, default=CONFIG["tdbb"]["f1_max"])
    parser.add_argument("--f2", type=float, default=CONFIG["tdbb"]["f2"])
    parser.add_argument("--target-scale", type=float, default=CONFIG["tdbb"]["target_scale"])
    parser.add_argument("--default-target", type=float, default=CONFIG["tdbb"]["default_target"])
    parser.add_argument("--calc-mode", default=CONFIG["pfp"]["calc_mode"])
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown command-line arguments: {unknown}")
    return args


def run_simulation(args: argparse.Namespace) -> None:
    input_path = resolve_existing_path(args.input, "Input structure", script_file=__file__)
    output_root = resolve_output_path(args.output_root)
    deepmd_root = output_root / CONFIG["output"]["deepmd_dir"]
    cif_root = output_root / CONFIG["output"]["cif_dir"]
    restart_root = output_root / CONFIG["output"]["restart_dir"]
    cif_root.mkdir(parents=True, exist_ok=True)
    restart_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / args.template_progress

    system = CONFIG["system"]
    relaxation = CONFIG["relaxation"]
    ctrl = CONFIG["md_control"]
    output = CONFIG["output"]

    params = TDBBParameters(
        gamma_kcal_mol_ps=args.gamma,
        f1_max_kcal_mol=args.f1_max,
        f2_inv_ang2=args.f2,
        target_scale=args.target_scale,
        default_target_ang=args.default_target,
    )
    base_calculator = build_pfp_calculator(args.calc_mode)

    if args.restart_from:
        atoms, restart_state = load_restart(
            args.restart_from,
            base_calculator,
            pbc=system["pbc"],
            fixed_z_lower=system["fixed_z_lower_bound"],
            fixed_z_upper=system["fixed_z_upper_bound"],
            script_file=__file__,
        )
        start_step = int(restart_state["step"])
    else:
        atoms = read(input_path)
        atoms.calc = base_calculator
        atoms.pbc = system["pbc"]
        relax_surface(
            atoms,
            fixed_z_lower=system["fixed_z_lower_bound"],
            fixed_z_upper=system["fixed_z_upper_bound"],
            surface_depth=system["surface_relax_depth"],
            fmax=relaxation["surface_fmax"],
        )
        relax_whole_structure(
            atoms,
            fixed_z_lower=system["fixed_z_lower_bound"],
            fixed_z_upper=system["fixed_z_upper_bound"],
            fmax=relaxation["whole_fmax"],
        )
        fixed = get_aimd_fixed_indices(
            atoms,
            system["fixed_z_lower_bound"],
            system["fixed_z_upper_bound"],
        )
        atoms.set_constraint(FixAtoms(indices=fixed))
        start_step = 0

    template_data = load_template_data(args, script_file=__file__) if args.pair_mode == "template" else None
    pairs = get_reactive_pairs(atoms, args)
    target_distances = get_template_target_distances(atoms, pairs, template_data, args) if template_data is not None else None
    target_keys = (
        [get_template_target_key(atoms, pair, template_data, args) for pair in pairs]
        if template_data is not None
        else ["auto"] * len(pairs)
    )

    bias = TDBBBias(atoms, pairs, params, target_distances=target_distances)
    atoms.calc = BiasedCalculator(base_calculator, bias)
    writer = DeepMDWriter(atoms, deepmd_root, set_interval=output["deepmd_set_interval"])

    if not args.restart_from:
        MaxwellBoltzmannDistribution(atoms, temperature_K=ctrl["initial_temp"])

    dyn = NVTBerendsen(
        atoms,
        timestep=ctrl["timestep"] * fs,
        temperature_K=ctrl["initial_temp"],
        taut=ctrl["tau_t"],
    )

    step_counter = {"step": start_step}
    run_state = {
        "phase": "not_started",
        "target_temperature": ctrl["initial_temp"],
    }

    def update_bias_time() -> None:
        bias.set_time(step_counter["step"] * ctrl["timestep"] / 1000.0)
        if atoms.calc is not None:
            atoms.calc.reset()

    if template_data is not None:
        distance_headers = [f"d_{i}_{j}_A" for i, j in pairs]
        target_headers = [f"target_{i}_{j}_A" for i, j in pairs]
        type_headers = [f"type_{i}_{j}" for i, j in pairs]
        if not progress_path.exists() or start_step == 0:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                ",".join(["step", "time_ps", "mean_abs_delta_A", *distance_headers, *target_headers, *type_headers]) + "\n"
            )

    def save_frame() -> None:
        step_counter["step"] += 1
        update_bias_time()

        step = step_counter["step"]
        if step % output["save_interval"] == 0:
            writer.add_frame(atoms, step)

            set_id = step // output["cif_set_interval"]
            cif_set_dir = cif_root / f"set_{set_id:03d}"
            cif_set_dir.mkdir(parents=True, exist_ok=True)
            write(cif_set_dir / f"step_{step:08d}.cif", atoms)

            if template_data is not None:
                distances = current_pair_distances(atoms, pairs)
                targets = np.asarray(target_distances, dtype=float)
                mean_abs_delta = float(np.mean(np.abs(distances - targets)))
                row = [
                    str(step),
                    f"{step * ctrl['timestep'] / 1000.0:.6f}",
                    f"{mean_abs_delta:.6f}",
                    *[f"{distance:.6f}" for distance in distances],
                    *[f"{target:.6f}" for target in targets],
                    *target_keys,
                ]
                with progress_path.open("a") as file:
                    file.write(",".join(row) + "\n")

        if step % output["restart_interval"] == 0:
            save_restart(
                atoms,
                restart_root,
                step,
                run_state["target_temperature"],
                run_state["phase"],
                timestep_fs=ctrl["timestep"],
                tau_t_fs=ctrl["tau_t"],
                input_file=args.input,
                calc_mode=args.calc_mode,
            )

    dyn.attach(save_frame, interval=1)

    print("Starting shared-helper TDBB AIMD workflow")
    print(f"Input: {input_path}")
    print(f"Pair mode: {args.pair_mode}")
    print(f"Reactive pairs: {pairs}")
    if args.pair_mode == "template":
        print(f"Template directory: {args.template_dir}")
        print(f"Template bond symbols: {args.template_bond_symbols}")
        print(f"Template pair source: {args.template_pair_source}")
        print(f"Template target mode: {args.template_target_mode}")
        print(f"Template progress: {args.template_progress}")
    print(f"Target distances (A): {[round(x, 3) for x in bias.target_distances]}")
    print(f"Target bond types: {list(zip(pairs, target_keys))}")
    print(f"PFP calc mode: {args.calc_mode}")
    print(f"Output root: {output_root}")

    completed_steps = start_step
    cumulative_steps = 0
    for segment in make_md_segments(
        initial_temp=ctrl["initial_temp"],
        final_temp=ctrl["final_temp"],
        ramp_interval=ctrl["ramp_interval"],
        ramp_steps=ctrl["ramp_steps"],
        stab_steps=ctrl["stab_steps"],
        prod_steps=ctrl["prod_steps"],
    ):
        segment_start = cumulative_steps
        segment_end = cumulative_steps + int(segment["steps"])
        cumulative_steps = segment_end

        if completed_steps >= segment_end:
            continue

        remaining_steps = segment_end - max(completed_steps, segment_start)
        run_state["phase"] = str(segment["phase"])
        run_state["target_temperature"] = float(segment["temperature"])
        dyn.set_temperature(temperature_K=float(segment["temperature"]))

        print(
            f"{segment['phase']} at {segment['temperature']} K: "
            f"run {remaining_steps} steps "
            f"(global {step_counter['step']} -> {segment_end})"
        )
        dyn.run(remaining_steps)
        completed_steps = segment_end


def main() -> None:
    run_simulation(parse_args())


if __name__ == "__main__":
    main()
