"""NEB-only entry point for manually selected endpoint structures.

Prepare the selected endpoint files under ``input/``:
- ``neb_selected_state0.cif``
- ``neb_selected_state1.cif``
- ``neb_selected_state2.cif``

This script skips adsorption-energy and AIMD screening steps. It only runs:
1. segment 0: state0 -> state1
2. segment 1: state1 -> state2
3. TS optimization, vibration analysis, and pseudo-IRC for each segment
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ase import Atoms

from freeenergy_neb.neb_from_notebook import (
    AimdConfig,
    NebConfig,
    RelaxConfig,
    StructureConfig,
    VibConfig,
    WorkflowConfig,
    build_runtime_config_from_relaxed_states,
    load_structure,
    prepare_neb_pairs_from_states,
    run_single_segment,
)


CONFIG = WorkflowConfig(
    compute_adsorption_energy=False,
    only_relax_states=False,
    structure=StructureConfig(
        input_dir=Path("input"),
        molecule_file="molecule_step0.cif",
        state_files=["neb_selected_state0.cif", "neb_selected_state1.cif", "neb_selected_state2.cif"],
        output_dir=Path("output"),
        structures_dir=Path("structures"),
        adsorbate_dir=Path("ad_structures"),
        neb_workdir="reaction_path_manual",
    ),
    relax=RelaxConfig(
        fix_z_max=1.0,
        relax_slab=False,
        relax_molecule=False,
        relax_adsorbate=False,
        relax_is_fs=True,
        do_coarse_relax_before_aimd=False,
        is_fmax=0.05,
        fs_fmax=0.005,
        ts_fmax=0.05,
        irc_fmax=0.005,
        irc_bfgs_steps=500,
        coarse_relax_fmax=0.2,
    ),
    neb=NebConfig(
        beads=21,
        spring_constant=0.05,
        parallel=True,
        climb=True,
        allow_shared_calculator=False,
        first_fmax=0.1,
        first_steps=2000,
        second_fmax=0.05,
        second_steps=10000,
        ts_index=11,
        segment_ts_indices=[10, 12],
    ),
    vib=VibConfig(
        z_threshold=7.0,
        mode_index=0,
        temperature=298.15,
        pressure=101325.0,
        geometry="linear",
        symmetrynumber=2,
        spin=0,
    ),
    aimd=AimdConfig(
        use_aimd_presampling=False,
    ),
)


def load_selected_states(config: WorkflowConfig = CONFIG) -> list[tuple[Path, Atoms]]:
    """Load manually chosen endpoint states without AIMD screening."""
    selected_states: list[tuple[Path, Atoms]] = []
    for state_file in config.structure.state_files:
        path = config.structure.input_dir / state_file
        atoms = load_structure(path)
        selected_states.append((path, atoms))
    return selected_states


def run_neb_only(
    config: WorkflowConfig = CONFIG,
    selected_states: Sequence[tuple[Path, Atoms]] | None = None,
) -> None:
    """Run NEB and post-processing from manually selected endpoint states."""
    if selected_states is None:
        selected_states = load_selected_states(config)

    runtime_config = build_runtime_config_from_relaxed_states(config, selected_states)
    segment_pairs = prepare_neb_pairs_from_states(runtime_config)
    for segment_index, (filepath, _, _) in enumerate(segment_pairs):
        print(f"Running NEB segment: {filepath}")
        run_single_segment(filepath, runtime_config, segment_index)


def main() -> None:
    run_neb_only(CONFIG)


if __name__ == "__main__":
    main()
