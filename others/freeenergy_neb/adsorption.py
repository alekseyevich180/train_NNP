"""Adsorption-energy-only entry point using already generated screened states.

Default inputs:
- ``input/molecule_step0.cif``
- ``input/slab_step0.cif``
- ``input/slab_step1.cif``
- ``input/slab_step2.cif``
- ``reaction_path_relaxed_states/neb_state0.cif``
- ``reaction_path_relaxed_states/neb_state1.cif``
- ``reaction_path_relaxed_states/neb_state2.cif``
"""

from __future__ import annotations

from pathlib import Path

from freeenergy_neb.neb_from_notebook import (
    AdsorptionPairConfig,
    AimdConfig,
    NebConfig,
    RelaxConfig,
    StructureConfig,
    VibConfig,
    WorkflowConfig,
    run_adsorption_energy_pairs,
)


CONFIG = WorkflowConfig(
    compute_adsorption_energy=True,
    only_relax_states=False,
    run_neb=False,
    adsorption_pairs=[
        AdsorptionPairConfig(
            name="step0",
            slab_file="slab_step0.cif",
            adsorbed_file="adsorbed_step0.cif",
            adsorbed_from_state="neb_state0.cif",
        ),
        AdsorptionPairConfig(
            name="step1",
            slab_file="slab_step1.cif",
            adsorbed_file="adsorbed_step1.cif",
            adsorbed_from_state="neb_state1.cif",
        ),
        AdsorptionPairConfig(
            name="step2",
            slab_file="slab_step2.cif",
            adsorbed_file="adsorbed_step2.cif",
            adsorbed_from_state="neb_state2.cif",
        ),
    ],
    structure=StructureConfig(
        input_dir=Path("input"),
        molecule_file="molecule_step0.cif",
        state_files=["neb_state0.cif", "neb_state1.cif", "neb_state2.cif"],
        output_dir=Path("output"),
        structures_dir=Path("structures"),
        adsorbate_dir=Path("ad_structures"),
        neb_workdir="reaction_path",
    ),
    relax=RelaxConfig(
        fix_z_max=1.0,
        relax_slab=True,
        relax_molecule=True,
        relax_adsorbate=True,
        relax_is_fs=False,
        do_coarse_relax_before_aimd=False,
        slab_fmax=0.005,
        molecule_fmax=0.005,
        adsorbate_fmax=0.005,
    ),
    neb=NebConfig(),
    vib=VibConfig(),
    aimd=AimdConfig(use_aimd_presampling=False),
)


def run_adsorption_only(config: WorkflowConfig = CONFIG) -> list[tuple[str, float]]:
    relaxed_state_dir = Path(f"{config.structure.neb_workdir}_relaxed_states")
    return run_adsorption_energy_pairs(config, relaxed_state_dir=relaxed_state_dir)


def main() -> None:
    run_adsorption_only(CONFIG)


if __name__ == "__main__":
    main()
