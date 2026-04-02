"""Standalone adsorption-energy + NEB workflow.

Expected files:
- input/molecule_step0.cif
- input/slab_step0.cif
- input/slab_step1.cif
- input/slab_step2.cif
- reaction_path_relaxed_states/neb_state0.cif
- reaction_path_relaxed_states/neb_state1.cif
- reaction_path_relaxed_states/neb_state2.cif
- reaction_path_manual_00_neb_state0_to_neb_state1/IS.cif
- reaction_path_manual_00_neb_state0_to_neb_state1/FS.cif
- reaction_path_manual_01_neb_state1_to_neb_state2/IS.cif
- reaction_path_manual_01_neb_state1_to_neb_state2/FS.cif
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory, read, write
from ase.mep import NEB
from ase.optimize import BFGS, FIRE
from ase.thermochemistry import IdealGasThermo
from ase.units import kB
from ase.vibrations import Vibrations
from PIL import Image
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator
from sella import Sella


@dataclass
class AdsorptionPairConfig:
    name: str
    slab_file: str
    adsorbed_state_file: str


@dataclass
class StructureConfig:
    input_dir: Path = Path("input")
    screened_state_dir: Path = Path("reaction_path_relaxed_states")
    molecule_file: str = "molecule_step0.cif"
    adsorption_pairs: list[AdsorptionPairConfig] = field(
        default_factory=lambda: [
            AdsorptionPairConfig("step0", "slab_step0.cif", "neb_state0.cif"),
            AdsorptionPairConfig("step1", "slab_step1.cif", "neb_state1.cif"),
            AdsorptionPairConfig("step2", "slab_step2.cif", "neb_state2.cif"),
        ]
    )
    neb_state_files: list[str] = field(
        default_factory=lambda: ["neb_state0.cif", "neb_state1.cif", "neb_state2.cif"]
    )
    neb_segment_dirs: list[Path] = field(
        default_factory=lambda: [
            Path("reaction_path_manual_00_neb_state0_to_neb_state1"),
            Path("reaction_path_manual_01_neb_state1_to_neb_state2"),
        ]
    )
    output_dir: Path = Path("output")
    structures_dir: Path = Path("structures")
    adsorbate_dir: Path = Path("ad_structures")
    neb_workdir: str = "reaction_path_manual"


@dataclass
class RelaxConfig:
    fix_z_max: float = 1.0
    relax_slab: bool = True
    relax_molecule: bool = True
    relax_adsorbate: bool = True
    relax_is_fs: bool = True
    slab_fmax: float = 0.005
    molecule_fmax: float = 0.005
    adsorbate_fmax: float = 0.005
    is_fmax: float = 0.05
    fs_fmax: float = 0.005
    ts_fmax: float = 0.05
    irc_fmax: float = 0.005
    irc_bfgs_steps: int = 500


@dataclass
class NebConfig:
    beads: int = 21
    spring_constant: float = 0.05
    parallel: bool = True
    climb: bool = True
    allow_shared_calculator: bool = False
    first_fmax: float = 0.1
    first_steps: int = 2000
    second_fmax: float = 0.05
    second_steps: int = 10000
    ts_index: int = 11
    segment_ts_indices: list[int] = field(default_factory=lambda: [10, 12])


@dataclass
class VibConfig:
    z_threshold: float = 7.0
    mode_index: int = 0
    temperature: float = 298.15
    pressure: float = 101325.0
    geometry: str = "linear"
    symmetrynumber: int = 2
    spin: int = 0
    mode_kT: float = 300 * kB
    mode_images: int = 30


@dataclass
class WorkflowConfig:
    calc_mode: str = "CRYSTAL"
    run_adsorption: bool = True
    run_neb: bool = True
    structure: StructureConfig = field(default_factory=StructureConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    neb: NebConfig = field(default_factory=NebConfig)
    vib: VibConfig = field(default_factory=VibConfig)


CONFIG = WorkflowConfig()


def build_calculator(calc_mode: str = "CRYSTAL") -> ASECalculator:
    estimator = Estimator(calc_mode=calc_mode)
    return ASECalculator(estimator)


calculator = build_calculator(CONFIG.calc_mode)


def load_structure(path: Path | str) -> Atoms:
    atoms = read(str(path))
    print(f"Loaded {path}: {len(atoms)} atoms")
    return atoms


def fixed_bottom_layer(atoms: Atoms, z_max: float) -> FixAtoms:
    return FixAtoms(indices=[atom.index for atom in atoms if atom.position[2] <= z_max])


def relax_structure(
    atoms: Atoms,
    fmax: float,
    trajectory: Optional[str],
    fix_bottom: bool,
    z_max: float,
) -> Atoms:
    atoms.calc = calculator
    if fix_bottom:
        atoms.set_constraint(fixed_bottom_layer(atoms, z_max))
    else:
        atoms.set_constraint()
    BFGS(atoms, trajectory=trajectory, logfile=None).run(fmax=fmax)
    return atoms


def load_slab_from_file(config: WorkflowConfig, path: Path | str, output_name: str) -> tuple[Atoms, float]:
    os.makedirs(config.structure.output_dir, exist_ok=True)
    os.makedirs(config.structure.structures_dir, exist_ok=True)
    slab = load_structure(path)
    if config.relax.relax_slab:
        relax_structure(
            slab,
            fmax=config.relax.slab_fmax,
            trajectory=str(config.structure.output_dir / f"{output_name}_opt.traj"),
            fix_bottom=True,
            z_max=config.relax.fix_z_max,
        )
    else:
        slab.calc = calculator
        slab.set_constraint(fixed_bottom_layer(slab, config.relax.fix_z_max))
    energy = slab.get_potential_energy()
    write(str(config.structure.structures_dir / f"{output_name}_relaxed.xyz"), slab)
    return slab, energy


def load_molecule_from_file(config: WorkflowConfig, path: Optional[Path | str] = None) -> tuple[Atoms, float]:
    os.makedirs(config.structure.output_dir, exist_ok=True)
    molecule = load_structure(path or (config.structure.input_dir / config.structure.molecule_file))
    if config.relax.relax_molecule:
        relax_structure(
            molecule,
            fmax=config.relax.molecule_fmax,
            trajectory=str(config.structure.output_dir / "molec_opt.traj"),
            fix_bottom=False,
            z_max=config.relax.fix_z_max,
        )
    else:
        molecule.calc = calculator
        molecule.set_constraint()
    return molecule, molecule.get_potential_energy()


def load_adsorbate_from_file(config: WorkflowConfig, path: Path | str, output_name: str) -> tuple[Atoms, float]:
    os.makedirs(config.structure.output_dir, exist_ok=True)
    os.makedirs(config.structure.adsorbate_dir, exist_ok=True)
    adsorbate = load_structure(path)
    if config.relax.relax_adsorbate:
        relax_structure(
            adsorbate,
            fmax=config.relax.adsorbate_fmax,
            trajectory=str(config.structure.output_dir / f"{output_name}_opt.traj"),
            fix_bottom=True,
            z_max=config.relax.fix_z_max,
        )
    else:
        adsorbate.calc = calculator
        adsorbate.set_constraint(fixed_bottom_layer(adsorbate, config.relax.fix_z_max))
    energy = adsorbate.get_potential_energy()
    write(str(config.structure.adsorbate_dir / f"{output_name}_relaxed.cif"), adsorbate)
    return adsorbate, energy


def run_adsorption_only(config: WorkflowConfig = CONFIG) -> list[tuple[str, float]]:
    molecule_path = config.structure.input_dir / config.structure.molecule_file
    _, e_mol = load_molecule_from_file(config, molecule_path)
    results: list[tuple[str, float]] = []

    for pair in config.structure.adsorption_pairs:
        slab_path = config.structure.input_dir / pair.slab_file
        adsorbate_path = config.structure.screened_state_dir / pair.adsorbed_state_file
        _, e_slab = load_slab_from_file(config, slab_path, output_name=f"{pair.name}_slab")
        _, e_ads = load_adsorbate_from_file(config, adsorbate_path, output_name=f"{pair.name}_adsorbed")
        adsorption_energy = e_ads - e_slab - e_mol
        print(f"Adsorption Energy [{pair.name}]: {adsorption_energy} eV")
        results.append((pair.name, adsorption_energy))

    return results


def get_segment_ts_index(config: WorkflowConfig, segment_index: int) -> int:
    if segment_index < len(config.neb.segment_ts_indices):
        return config.neb.segment_ts_indices[segment_index]
    return config.neb.ts_index


def get_existing_neb_segments(config: WorkflowConfig = CONFIG) -> list[Path]:
    if not config.structure.neb_segment_dirs:
        raise ValueError("config.structure.neb_segment_dirs is empty.")
    missing = [path for path in config.structure.neb_segment_dirs if not (path / "IS.cif").exists() or not (path / "FS.cif").exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing IS.cif or FS.cif in segment directories: {missing_text}")
    return config.structure.neb_segment_dirs


def calc_max_force(atoms: Atoms) -> float:
    return ((atoms.get_forces() ** 2).sum(axis=1).max()) ** 0.5


def run_neb(filepath: str, config: WorkflowConfig) -> list[Atoms]:
    is_atoms = load_structure(f"{filepath}/IS.cif")
    fs_atoms = load_structure(f"{filepath}/FS.cif")

    is_atoms.set_constraint(fixed_bottom_layer(is_atoms, config.relax.fix_z_max))
    is_atoms.calc = calculator
    BFGS(is_atoms, logfile=None).run(fmax=config.relax.is_fmax)

    fs_atoms.set_constraint(fixed_bottom_layer(fs_atoms, config.relax.fix_z_max))
    fs_atoms.calc = calculator
    BFGS(fs_atoms, logfile=None).run(fmax=config.relax.fs_fmax)

    configs = [is_atoms.copy() for _ in range(config.neb.beads - 1)] + [fs_atoms.copy()]
    for image in configs:
        image.calc = build_calculator(config.calc_mode)

    neb = NEB(
        configs,
        k=config.neb.spring_constant,
        parallel=config.neb.parallel,
        climb=config.neb.climb,
        allow_shared_calculator=config.neb.allow_shared_calculator,
    )
    neb.interpolate()
    relax = FIRE(neb, trajectory=None, logfile=f"{filepath}/neb_log.txt")
    relax.run(fmax=config.neb.first_fmax, steps=config.neb.first_steps)
    relax.run(fmax=config.neb.second_fmax, steps=config.neb.second_steps)
    write(f"{filepath}/NEB_images.xyz", configs)
    return configs


def analyze_neb(filepath: str, config: WorkflowConfig, ts_index: int) -> tuple[list[Atoms], list[float], list[float]]:
    configs = read(f"{filepath}/NEB_images.xyz", index=":")
    for image in configs:
        image.calc = build_calculator(config.calc_mode)

    energies = [image.get_total_energy() for image in configs]
    mforces = [calc_max_force(image) for image in configs]
    profile = pd.DataFrame(
        {
            "image_index": list(range(len(configs))),
            "energy_eV": energies,
            "max_force_eVA": mforces,
        }
    )
    profile.to_csv(f"{filepath}/neb_profile.csv", index=False)

    plt.figure()
    plt.plot(range(len(energies)), energies, marker="o")
    plt.xlabel("replica")
    plt.ylabel("energy [eV]")
    plt.xticks(np.arange(0, len(energies), 2))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filepath}/neb_energy_profile.png", dpi=200)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(range(len(mforces)), mforces, marker="o")
    plt.xlabel("replica")
    plt.ylabel("max force [eV/A]")
    plt.xticks(np.arange(0, len(mforces), 2))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filepath}/neb_force_profile.png", dpi=200)
    plt.show()
    plt.close()

    print(f"actE {energies[ts_index] - energies[0]} eV, deltaE {energies[ts_index] - energies[-1]} eV")
    return configs, energies, mforces


def optimize_ts(filepath: str, configs: Sequence[Atoms], config: WorkflowConfig, ts_index: int) -> tuple[Atoms, pd.DataFrame]:
    ts = configs[ts_index].copy()
    ts.set_constraint(fixed_bottom_layer(ts, config.relax.fix_z_max))
    z_pos = pd.DataFrame({"symbol": ts.get_chemical_symbols(), "z": ts.get_positions()[:, 2]})
    ts.calc = calculator
    Sella(ts).run(fmax=config.relax.ts_fmax)
    write(f"{filepath}/TS_opt.cif", ts)
    return ts, z_pos


def run_vibrational_analysis(
    filepath: str,
    ts: Atoms,
    z_pos: pd.DataFrame,
    config: WorkflowConfig,
):
    vibatoms = z_pos[z_pos["z"] >= config.vib.z_threshold].index
    vibpath = f"{filepath}/TS_vib/vib"
    os.makedirs(vibpath, exist_ok=True)
    vib = Vibrations(ts, name=vibpath, indices=vibatoms)
    vib.run()
    vib_energies = vib.get_energies()
    n_imag = int(np.sum(np.iscomplex(vib_energies)))
    if n_imag:
        print(f"Found {n_imag} imaginary vibrational mode(s); skipping Gibbs free-energy evaluation for TS.")
    else:
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            potentialenergy=ts.get_potential_energy(),
            atoms=ts,
            geometry=config.vib.geometry,
            symmetrynumber=config.vib.symmetrynumber,
            spin=config.vib.spin,
            natoms=len(vibatoms),
        )
        gibbs = thermo.get_gibbs_energy(
            temperature=config.vib.temperature,
            pressure=config.vib.pressure,
        )
        print(f"TS Gibbs energy: {gibbs} eV")
    vib.summary(log=f"{filepath}/vib_summary.txt")
    vib.write_mode(n=config.vib.mode_index, kT=config.vib.mode_kT, nimages=config.vib.mode_images)
    vib.clean()
    vib_traj = Trajectory(f"{vibpath}.{config.vib.mode_index}.traj")
    write(f"{filepath}/vib_traj.xyz", vib_traj)
    return read(f"{filepath}/vib_traj.xyz", index=":")


def pseudo_irc(ts: Atoms, vib_traj: Sequence[Atoms], config: WorkflowConfig) -> tuple[Atoms, Atoms]:
    constraint = fixed_bottom_layer(ts, config.relax.fix_z_max)

    irc_is = vib_traj[14].copy()
    irc_is.calc = calculator
    irc_is.set_constraint(constraint)
    BFGS(irc_is, logfile=None, maxstep=0.5).run(
        fmax=config.relax.irc_fmax, steps=config.relax.irc_bfgs_steps
    )

    irc_fs = vib_traj[16].copy()
    irc_fs.calc = calculator
    irc_fs.set_constraint(constraint)
    BFGS(irc_fs, logfile=None, maxstep=0.5).run(
        fmax=config.relax.irc_fmax, steps=config.relax.irc_bfgs_steps
    )
    return irc_is, irc_fs


def render_neb_images(filepath: str) -> None:
    configs = read(f"{filepath}/NEB_images.xyz", index=":")
    os.makedirs(f"{filepath}/png_NEB", exist_ok=True)
    for i, atoms in enumerate(configs):
        write(f"{filepath}/png_NEB/NEB_{i:03}.png", atoms.copy(), rotation="-60x, 30y, 15z")

    images = []
    for image_path in sorted(glob.glob(f"{filepath}/png_NEB/*.png")):
        image = Image.open(image_path)
        image.load()
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        images.append(background)
    if images:
        images[0].save(
            f"{filepath}/gif_NEB.gif",
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=100,
            loop=0,
        )


def run_single_segment(filepath: str, config: WorkflowConfig, segment_index: int) -> None:
    ts_index = get_segment_ts_index(config, segment_index)
    run_neb(filepath, config)
    configs, _, _ = analyze_neb(filepath, config, ts_index)
    render_neb_images(filepath)
    ts, z_pos = optimize_ts(filepath, configs, config, ts_index)
    vib_traj = run_vibrational_analysis(filepath, ts, z_pos, config)
    irc_is, irc_fs = pseudo_irc(ts, vib_traj, config)
    write(f"{filepath}/IS_IRC.xyz", irc_is)
    write(f"{filepath}/FS_IRC.xyz", irc_fs)


def run_neb_only(config: WorkflowConfig = CONFIG) -> None:
    segment_dirs = get_existing_neb_segments(config)
    for segment_index, segment_dir in enumerate(segment_dirs):
        print(f"Running NEB segment: {segment_dir}")
        run_single_segment(str(segment_dir), config, segment_index)


def run_all(config: WorkflowConfig = CONFIG) -> None:
    if config.run_adsorption:
        run_adsorption_only(config)
    if config.run_neb:
        run_neb_only(config)


def main() -> None:
    run_all(CONFIG)


if __name__ == "__main__":
    main()
