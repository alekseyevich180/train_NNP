"""NEB workflow extracted from 1_Solid_cat_NEB_TSopt_example_ja.ipynb.

Expected input files under ``input/``:
- ``molecule_step0.cif``: isolated molecular reference for adsorption energies
- ``slab_step0.cif`` / ``adsorbed_step0.cif``: adsorption pair for step 0
- ``slab_step1.cif`` / ``adsorbed_step1.cif``: adsorption pair for step 1
- ``slab_step2.cif`` / ``adsorbed_step2.cif``: adsorption pair for step 2
- ``neb_state0.cif`` / ``neb_state1.cif`` / ``neb_state2.cif``: NEB endpoint states
"""

from __future__ import annotations

import glob
import os
import threading
import time
from dataclasses import dataclass, field
from math import pi
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import nglview as nv
import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import add_adsorbate, molecule, sort, surface
from ase.constraints import ExpCellFilter, FixAtoms, FixBondLengths
from ase.io import Trajectory, read, write
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.mep import NEB
from ase.optimize import BFGS, FIRE, LBFGS
from ase.optimize.basin import BasinHopping
from ase.optimize.minimahopping import MinimaHopping
from ase.thermochemistry import IdealGasThermo
from ase.units import fs, kB
from ase.vibrations import Vibrations
from ase.visualize import view
from IPython.display import Image as ImageWidget
from IPython.display import display
from ipywidgets import (
    Button,
    Checkbox,
    FloatSlider,
    GridspecLayout,
    HBox,
    IntSlider,
    Label,
    Text,
    Textarea,
)
from nglview.widget import NGLWidget
from PIL import Image
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator
from sella import Sella


@dataclass
class StructureConfig:
    input_dir: Path = Path("input")
    molecule_file: str = "molecule_step0.cif"
    is_file: str = "IS.cif"
    fs_file: str = "FS.cif"
    state_files: list[str] = field(default_factory=lambda: ["neb_state0.cif", "neb_state1.cif", "neb_state2.cif"])
    output_dir: Path = Path("output")
    structures_dir: Path = Path("structures")
    adsorbate_dir: Path = Path("ad_structures")
    neb_workdir: str = "neb_run"


@dataclass
class AdsorptionPairConfig:
    name: str
    slab_file: str
    adsorbed_file: str
    adsorbed_from_state: Optional[str] = None


@dataclass
class RelaxConfig:
    fix_z_max: float = 1.0
    relax_slab: bool = True
    relax_molecule: bool = True
    relax_adsorbate: bool = True
    relax_is_fs: bool = True
    do_coarse_relax_before_aimd: bool = True
    slab_fmax: float = 0.005
    molecule_fmax: float = 0.005
    adsorbate_fmax: float = 0.005
    is_fmax: float = 0.05
    fs_fmax: float = 0.005
    ts_fmax: float = 0.05
    irc_fmax: float = 0.005
    irc_bfgs_steps: int = 500
    coarse_relax_fmax: float = 0.2


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
    segment_ts_indices: list[int] = field(default_factory=list)


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
class AimdConfig:
    use_aimd_presampling: bool = True
    temperature_K: float = 300.0
    steps: int = 30000
    timestep_fs: float = 0.5
    tau_t: float = 100.0
    sample_interval: int = 100
    relax_sampled_frames: bool = True
    sampled_frame_fmax: float = 0.05


@dataclass
class WorkflowConfig:
    calc_mode: str = "CRYSTAL"
    compute_adsorption_energy: bool = False
    only_relax_states: bool = False
    run_neb: bool = True
    adsorption_pairs: list[AdsorptionPairConfig] = field(default_factory=list)
    structure: StructureConfig = field(default_factory=StructureConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    neb: NebConfig = field(default_factory=NebConfig)
    vib: VibConfig = field(default_factory=VibConfig)
    aimd: AimdConfig = field(default_factory=AimdConfig)

    @property
    def molecule_path(self) -> Path:
        return self.structure.input_dir / self.structure.molecule_file

    @property
    def is_path(self) -> Path:
        return self.structure.input_dir / self.structure.is_file

    @property
    def fs_path(self) -> Path:
        return self.structure.input_dir / self.structure.fs_file

    @property
    def state_paths(self) -> list[Path]:
        return [self.structure.input_dir / name for name in self.structure.state_files]


CONFIG = WorkflowConfig(
    compute_adsorption_energy=True,
    only_relax_states=False,
    run_neb=True,
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
        relax_molecule=True,
        relax_adsorbate=True,
        relax_is_fs=True,
        do_coarse_relax_before_aimd=True,
        molecule_fmax=0.005,
        adsorbate_fmax=0.005,
        is_fmax=0.05,
        fs_fmax=0.005,
        ts_fmax=0.05,
        coarse_relax_fmax=0.2,
    ),
    neb=NebConfig(
        beads=21,
        spring_constant=0.05,
        first_fmax=0.1,
        first_steps=2000,
        second_fmax=0.05,
        second_steps=10000,
        ts_index=11,
        segment_ts_indices=[10, 12],
    ),
    aimd=AimdConfig(
        use_aimd_presampling=True,
        temperature_K=300.0,
        steps=30000,
        timestep_fs=0.5,
        tau_t=100.0,
        sample_interval=100,
        relax_sampled_frames=True,
        sampled_frame_fmax=0.05,
    ),
)


def build_calculator(calc_mode: str = "CRYSTAL") -> ASECalculator:
    estimator = Estimator(calc_mode=calc_mode)
    return ASECalculator(estimator)


calculator = build_calculator(CONFIG.calc_mode)


def myopt(
    atoms: Atoms,
    sn: int = 10,
    constraintatoms: Optional[Sequence[int]] = None,
    cbonds: Optional[Sequence[Sequence[int]]] = None,
) -> Atoms:
    constraintatoms = list(constraintatoms or [])
    cbonds = list(cbonds or [])
    fa = FixAtoms(indices=constraintatoms)
    fb = FixBondLengths(cbonds, tolerance=1e-5)
    atoms.set_constraint([fa, fb])
    atoms.calc = calculator
    maxf = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
    print(f"ini   pot:{atoms.get_potential_energy():.4f},maxforce:{maxf:.4f}")
    de = -1.0
    s = 1
    ita = 50
    while (de < -0.001 or de > 0.001) and s <= sn:
        opt = BFGS(atoms, maxstep=0.04 * (0.9**s), logfile=None)
        old = atoms.get_potential_energy()
        opt.run(fmax=0.0005, steps=ita)
        maxf = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
        de = atoms.get_potential_energy() - old
        print(f"{s * ita} pot:{atoms.get_potential_energy():.4f},maxforce:{maxf:.4f},delta:{de:.4f}")
        s += 1
    return atoms


def opt_cell_size(atoms: Atoms, sn: int = 10, iter_count: bool = False):
    atoms.set_constraint()
    atoms.calc = calculator
    maxf = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
    ucf = ExpCellFilter(atoms)
    print(f"ini   pot:{atoms.get_potential_energy():.4f},maxforce:{maxf:.4f}")
    de = -1.0
    s = 1
    ita = 50
    while (de < -0.01 or de > 0.01) and s <= sn:
        opt = BFGS(ucf, maxstep=0.04 * (0.9**s), logfile=None)
        old = atoms.get_potential_energy()
        opt.run(fmax=0.005, steps=ita)
        maxf = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
        de = atoms.get_potential_energy() - old
        print(f"{s * ita} pot:{atoms.get_potential_energy():.4f},maxforce:{maxf:.4f},delta:{de:.4f}")
        s += 1
    if iter_count:
        return atoms, s * ita
    return atoms


def makesurface(
    atoms: Atoms,
    miller_indices: Sequence[int] = (1, 1, 1),
    layers: int = 4,
    rep: Sequence[int] = (4, 4, 1),
) -> Atoms:
    slab = surface(atoms, miller_indices, layers)
    slab.center(vacuum=10.0, axis=2)
    slab = slab.repeat(rep)
    slab.set_positions(slab.get_positions() - [0, 0, min(slab.get_positions()[:, 2])])
    slab.pbc = True
    return slab


def save_image(filename: str, widget: NGLWidget) -> None:
    image = widget.render_image()
    while not image.value:
        time.sleep(0.1)
    with open(filename, "wb") as fh:
        fh.write(image.value)


class SurfaceEditor:
    """Structure viewer/editor copied from the notebook for manual operations."""

    struct: List[Dict]

    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.vh = view(atoms, viewer="ngl")
        self.v: NGLWidget = self.vh.children[0]
        self.v._remote_call("setSize", args=["450px", "450px"])
        self.recont()
        self.set_representation()
        self.set_atoms()
        self.pots: List[float] = []
        self.traj: List[Atoms] = []
        self.cal_nnp()

    def display(self):
        display(self.vh)

    def recont(self):
        self.vh.setatoms = FloatSlider(min=0, max=50, step=0.1, value=8, description="atoms z>")
        self.vh.setatoms.observe(self.set_atoms)
        self.vh.selected_atoms_label = Label("Selected atoms:")
        self.vh.selected_atoms_textarea = Textarea()
        selected_atoms_hbox = HBox([self.vh.selected_atoms_label, self.vh.selected_atoms_textarea])
        self.vh.move = FloatSlider(min=0.1, max=2, step=0.1, value=0.5, description="move")

        grid1 = GridspecLayout(2, 3)
        self.vh.xplus = Button(description="X+")
        self.vh.xminus = Button(description="X-")
        self.vh.yplus = Button(description="Y+")
        self.vh.yminus = Button(description="Y-")
        self.vh.zplus = Button(description="Z+")
        self.vh.zminus = Button(description="Z-")
        self.vh.xplus.on_click(self.move)
        self.vh.xminus.on_click(self.move)
        self.vh.yplus.on_click(self.move)
        self.vh.yminus.on_click(self.move)
        self.vh.zplus.on_click(self.move)
        self.vh.zminus.on_click(self.move)
        grid1[0, 0] = self.vh.xplus
        grid1[0, 1] = self.vh.yplus
        grid1[0, 2] = self.vh.zplus
        grid1[1, 0] = self.vh.xminus
        grid1[1, 1] = self.vh.yminus
        grid1[1, 2] = self.vh.zminus

        self.vh.rotate = FloatSlider(min=1, max=90, step=1, value=30, description="rotate")
        grid2 = GridspecLayout(2, 3)
        self.vh.xplus2 = Button(description="X+")
        self.vh.xminus2 = Button(description="X-")
        self.vh.yplus2 = Button(description="Y+")
        self.vh.yminus2 = Button(description="Y-")
        self.vh.zplus2 = Button(description="Z+")
        self.vh.zminus2 = Button(description="Z-")
        self.vh.xplus2.on_click(self.rotate)
        self.vh.xminus2.on_click(self.rotate)
        self.vh.yplus2.on_click(self.rotate)
        self.vh.yminus2.on_click(self.rotate)
        self.vh.zplus2.on_click(self.rotate)
        self.vh.zminus2.on_click(self.rotate)
        grid2[0, 0] = self.vh.xplus2
        grid2[0, 1] = self.vh.yplus2
        grid2[0, 2] = self.vh.zplus2
        grid2[1, 0] = self.vh.xminus2
        grid2[1, 1] = self.vh.yminus2
        grid2[1, 2] = self.vh.zminus2

        self.vh.nnptext = Textarea(disabled=True)
        self.vh.opt_step = IntSlider(min=0, max=100, step=1, value=10, description="Opt steps")
        self.vh.constraint_checkbox = Checkbox(value=True, description="Opt only selected atoms")
        self.vh.run_opt_button = Button(
            description="Run mini opt",
            tooltip="Execute BFGS optimization with small step update.",
        )
        self.vh.run_opt_button.on_click(self.run_opt)
        opt_hbox = HBox([self.vh.constraint_checkbox, self.vh.run_opt_button])

        self.vh.filename_text = Text(value="screenshot.png", description="filename: ")
        self.vh.download_image_button = Button(description="download image")
        self.vh.download_image_button.on_click(self.download_image)
        self.vh.save_image_button = Button(description="save image")
        self.vh.save_image_button.on_click(self.save_image)

        self.vh.update_display = Button(description="update_display")
        self.vh.update_display.on_click(self.update_display)

        controls = list(self.vh.control_box.children)
        controls += [
            self.vh.setatoms,
            selected_atoms_hbox,
            self.vh.move,
            grid1,
            self.vh.rotate,
            grid2,
            self.vh.nnptext,
            self.vh.opt_step,
            opt_hbox,
            self.vh.filename_text,
            HBox([self.vh.download_image_button, self.vh.save_image_button]),
            self.vh.update_display,
        ]
        self.vh.control_box.children = tuple(controls)

    def set_representation(self, bcolor: str = "white"):
        self.v.background = bcolor
        self.struct = self.get_struct(self.atoms)
        self.v.add_representation(repr_type="ball+stick")
        self.v.control.spin([0, 1, 0], pi * 1.1)
        self.v.control.spin([1, 0, 0], -pi * 0.45)
        threading.Thread(target=self.changestr).start()

    def changestr(self):
        time.sleep(2)
        self.v._remote_call("replaceStructure", target="Widget", args=self.struct)

    def get_struct(self, atoms: Atoms, ext: str = "pdb") -> List[Dict]:
        struct = nv.ASEStructure(atoms, ext=ext).get_structure_string()
        for c in range(len(atoms)):
            struct = struct.replace("MOL     1", "M0    " + str(c).zfill(3), 1)
        return [dict(data=struct, ext=ext)]

    def cal_nnp(self):
        pot = self.atoms.get_potential_energy()
        mforce = (((self.atoms.get_forces()) ** 2).sum(axis=1).max()) ** 0.5
        self.vh.nnptext.value = f"pot energy: {pot} eV\nmax force : {mforce} eV/A"
        self.pots.append(pot)
        self.traj.append(self.atoms.copy())

    def update_display(self, clicked_button: Optional[Button] = None):
        _ = clicked_button
        struct = self.get_struct(self.atoms)
        self.struct = struct
        self.v._remote_call("replaceStructure", target="Widget", args=struct)
        self.cal_nnp()

    def set_atoms(self, slider: Optional[FloatSlider] = None):
        _ = slider
        selected = [i for i, atom in enumerate(self.atoms) if atom.z >= self.vh.setatoms.value]
        self.vh.selected_atoms_textarea.value = ", ".join(map(str, selected))

    def get_selected_atom_indices(self) -> List[int]:
        values = self.vh.selected_atoms_textarea.value.split(",")
        return [int(a) for a in values if a.strip()]

    def move(self, clicked_button: Button):
        delta = self.vh.move.value
        for index in self.get_selected_atom_indices():
            if clicked_button.description == "X+":
                self.atoms[index].position += [delta, 0, 0]
            elif clicked_button.description == "X-":
                self.atoms[index].position -= [delta, 0, 0]
            elif clicked_button.description == "Y+":
                self.atoms[index].position += [0, delta, 0]
            elif clicked_button.description == "Y-":
                self.atoms[index].position -= [0, delta, 0]
            elif clicked_button.description == "Z+":
                self.atoms[index].position += [0, 0, delta]
            elif clicked_button.description == "Z-":
                self.atoms[index].position -= [0, 0, delta]
        self.update_display()

    def rotate(self, clicked_button: Button):
        atom_indices = self.get_selected_atom_indices()
        deg = self.vh.rotate.value
        temp = self.atoms[atom_indices]
        if clicked_button.description == "X+":
            temp.rotate(deg, "x", center="COP")
        elif clicked_button.description == "X-":
            temp.rotate(-deg, "x", center="COP")
        elif clicked_button.description == "Y+":
            temp.rotate(deg, "y", center="COP")
        elif clicked_button.description == "Y-":
            temp.rotate(-deg, "y", center="COP")
        elif clicked_button.description == "Z+":
            temp.rotate(deg, "z", center="COP")
        elif clicked_button.description == "Z-":
            temp.rotate(-deg, "z", center="COP")
        for i, atom in enumerate(atom_indices):
            self.atoms[atom].position = temp.positions[i]
        self.update_display()

    def run_opt(self, clicked_button: Button):
        _ = clicked_button
        if self.vh.constraint_checkbox.value:
            atom_indices = self.get_selected_atom_indices()
            constrained = [i for i in range(len(self.atoms)) if i not in atom_indices]
            self.atoms.set_constraint(FixAtoms(indices=constrained))
        opt = BFGS(self.atoms, maxstep=0.04, logfile=None)
        steps = self.vh.opt_step.value
        opt.run(fmax=0.0001, steps=steps)
        self.update_display()

    def download_image(self, clicked_button: Optional[Button] = None):
        _ = clicked_button
        self.v.download_image(filename=self.vh.filename_text.value)

    def save_image(self, clicked_button: Optional[Button] = None):
        _ = clicked_button
        filename = self.vh.filename_text.value
        if filename.endswith(".png"):
            threading.Thread(target=save_image, args=(filename, self.v), daemon=True).start()
        elif filename.endswith(".html"):
            nv.write_html(filename, [self.v])  # type: ignore[arg-type]
        else:
            print(f"filename {filename}: extension not supported!")


def fixed_bottom_layer(atoms: Atoms, z_max: float = 1.0) -> FixAtoms:
    return FixAtoms(indices=[atom.index for atom in atoms if atom.position[2] <= z_max])


def plot_z_positions(atoms: Atoms) -> pd.DataFrame:
    z_pos = pd.DataFrame({"symbol": atoms.get_chemical_symbols(), "z": atoms.get_positions()[:, 2]})
    plt.scatter(z_pos.index, z_pos["z"])
    plt.grid(True)
    plt.xlabel("atom_index")
    plt.ylabel("z_position")
    plt.show()
    return z_pos


def load_structure(path: Path | str) -> Atoms:
    atoms = read(str(path))
    print(f"Loaded {path}: {len(atoms)} atoms")
    return atoms


def reorder_atoms_like_reference(reference: Atoms, target: Atoms) -> Atoms:
    """Reorder target atoms to match reference ordering for NEB."""
    ref_numbers = reference.get_atomic_numbers()
    tgt_numbers = target.get_atomic_numbers()
    if len(ref_numbers) != len(tgt_numbers):
        raise ValueError("Reference and target have different atom counts.")
    if sorted(ref_numbers.tolist()) != sorted(tgt_numbers.tolist()):
        raise ValueError("Reference and target do not have the same composition.")

    ref_positions = reference.get_positions()
    tgt_positions = target.get_positions()
    remaining = list(range(len(target)))
    new_order: list[int] = []

    for i, atomic_number in enumerate(ref_numbers):
        candidates = [idx for idx in remaining if tgt_numbers[idx] == atomic_number]
        if not candidates:
            raise ValueError(f"Could not find matching atom for atomic number {atomic_number}.")
        distances = [np.linalg.norm(tgt_positions[idx] - ref_positions[i]) for idx in candidates]
        best_idx = candidates[int(np.argmin(distances))]
        new_order.append(best_idx)
        remaining.remove(best_idx)

    reordered = target[new_order]
    reordered.set_cell(target.cell)
    reordered.set_pbc(target.pbc)
    return reordered


def relax_structure(
    atoms: Atoms,
    config: WorkflowConfig,
    fmax: float = 0.005,
    trajectory: Optional[str] = None,
    fix_bottom: bool = False,
) -> Atoms:
    atoms.calc = calculator
    if fix_bottom:
        atoms.set_constraint(fixed_bottom_layer(atoms, z_max=config.relax.fix_z_max))
    else:
        atoms.set_constraint()
    BFGS(atoms, trajectory=trajectory, logfile=None).run(fmax=fmax)
    return atoms


def load_slab_from_file(
    config: WorkflowConfig,
    path: Path | str,
    output_name: str = "slab",
) -> tuple[Atoms, float]:
    os.makedirs(config.structure.output_dir, exist_ok=True)
    os.makedirs(config.structure.structures_dir, exist_ok=True)
    slab = load_structure(path)
    plot_z_positions(slab)
    print("highest position (z) =", slab.get_positions()[:, 2].max())
    if config.relax.relax_slab:
        relax_structure(
            slab,
            config,
            fmax=config.relax.slab_fmax,
            trajectory=str(config.structure.output_dir / f"{output_name}_opt.traj"),
            fix_bottom=True,
        )
    else:
        slab.calc = calculator
        slab.set_constraint(fixed_bottom_layer(slab, z_max=config.relax.fix_z_max))
    slab_energy = slab.get_potential_energy()
    print(f"slab E = {slab_energy} eV")
    write(str(config.structure.structures_dir / f"{output_name}_relaxed.xyz"), slab)
    return slab, slab_energy


def load_molecule_from_file(config: WorkflowConfig, path: Optional[Path | str] = None) -> tuple[Atoms, float]:
    os.makedirs(config.structure.output_dir, exist_ok=True)
    molec = load_structure(path or config.molecule_path)
    if config.relax.relax_molecule:
        relax_structure(
            molec,
            config,
            fmax=config.relax.molecule_fmax,
            trajectory=str(config.structure.output_dir / "molec_opt.traj"),
            fix_bottom=False,
        )
    else:
        molec.calc = calculator
        molec.set_constraint()
    molec_energy = molec.get_potential_energy()
    print(f"molecE = {molec_energy} eV")
    return molec, molec_energy


def load_adsorbate_from_file(
    config: WorkflowConfig,
    path: Path | str,
    output_name: str,
) -> tuple[Atoms, float]:
    adsorbate = load_structure(path)
    if config.relax.relax_adsorbate:
        relax_structure(
            adsorbate,
            config,
            fmax=config.relax.adsorbate_fmax,
            trajectory=str(config.structure.output_dir / f"{output_name}_opt.traj"),
            fix_bottom=True,
        )
    else:
        adsorbate.calc = calculator
        adsorbate.set_constraint(fixed_bottom_layer(adsorbate, z_max=config.relax.fix_z_max))
    adsorbate_energy = adsorbate.get_potential_energy()
    print(f"mol_on_slabE = {adsorbate_energy} eV")

    os.makedirs(config.structure.adsorbate_dir, exist_ok=True)
    write(str(config.structure.adsorbate_dir / f"{output_name}_relaxed.cif"), adsorbate)
    return adsorbate, adsorbate_energy


def adsorption_energy(slab_energy: float, molec_energy: float, mol_on_slab_energy: float, label: str) -> float:
    adsorp_energy = slab_energy + molec_energy - mol_on_slab_energy
    print(f"Adsorption Energy [{label}]: {adsorp_energy} eV")
    return adsorp_energy


def report_adsorption_energy_scheme() -> None:
    print("Adsorption energy uses: E_ads = E_slab + E_molecule - E_adsorbed")
    print("Each adsorbed structure must be paired with its corresponding slab reference.")
    print("When adsorbed_from_state is set, the screened NEB state is used as E_adsorbed.")
    print("For large molecules, make sure molecule_step0.cif is the isolated reference conformer you want to compare against.")


def run_adsorption_energy_pairs(
    config: WorkflowConfig,
    relaxed_state_dir: Optional[Path] = None,
) -> list[tuple[str, float]]:
    if not config.adsorption_pairs:
        raise ValueError("compute_adsorption_energy=True requires at least one adsorption pair.")

    _, molec_energy = load_molecule_from_file(config)
    results: list[tuple[str, float]] = []
    report_adsorption_energy_scheme()

    for pair in config.adsorption_pairs:
        slab_path = config.structure.input_dir / pair.slab_file
        if pair.adsorbed_from_state and relaxed_state_dir is not None:
            adsorbed_path = relaxed_state_dir / pair.adsorbed_from_state
            print(f"Using screened state for adsorption pair {pair.name}: {adsorbed_path}")
        else:
            adsorbed_path = config.structure.input_dir / pair.adsorbed_file
        _, slab_energy = load_slab_from_file(config, slab_path, output_name=f"{pair.name}_slab")
        _, adsorbed_energy = load_adsorbate_from_file(config, adsorbed_path, output_name=f"{pair.name}_adsorbed")
        results.append((pair.name, adsorption_energy(slab_energy, molec_energy, adsorbed_energy, pair.name)))

    return results


def load_adsorption_structures(ad_st_path: str = "ad_structures/*") -> list[tuple[str, Atoms]]:
    return [(filepath, read(filepath)) for filepath in glob.glob(ad_st_path)]


def prepare_is_fs_from_files(
    config: WorkflowConfig,
    is_file: Optional[Path | str] = None,
    fs_file: Optional[Path | str] = None,
    workdir: Optional[str] = None,
) -> tuple[str, Atoms, Atoms]:
    is_atoms = load_structure(is_file or config.is_path)
    fs_atoms = reorder_atoms_like_reference(is_atoms, load_structure(fs_file or config.fs_path))

    if config.relax.relax_is_fs:
        relax_structure(is_atoms, config, fmax=config.relax.is_fmax, fix_bottom=True)
        relax_structure(fs_atoms, config, fmax=config.relax.fs_fmax, fix_bottom=True)
    else:
        is_atoms.calc = calculator
        fs_atoms.calc = calculator
        is_atoms.set_constraint(fixed_bottom_layer(is_atoms, z_max=config.relax.fix_z_max))
        fs_atoms.set_constraint(fixed_bottom_layer(fs_atoms, z_max=config.relax.fix_z_max))

    target_workdir = workdir or config.structure.neb_workdir
    os.makedirs(target_workdir, exist_ok=True)
    write(f"{target_workdir}/IS.cif", is_atoms)
    write(f"{target_workdir}/FS.cif", fs_atoms)
    return target_workdir, is_atoms, fs_atoms


def prepare_neb_pairs_from_states(config: WorkflowConfig) -> list[tuple[str, Atoms, Atoms]]:
    state_paths = config.state_paths
    if len(state_paths) < 2:
        raise ValueError("At least two state files are required in config.structure.state_files.")

    pairs: list[tuple[str, Atoms, Atoms]] = []
    for i in range(len(state_paths) - 1):
        initial_path = state_paths[i]
        final_path = state_paths[i + 1]
        workdir = f"{config.structure.neb_workdir}_{i:02d}_{initial_path.stem}_to_{final_path.stem}"
        pairs.append(
            prepare_is_fs_from_files(
                config,
                is_file=initial_path,
                fs_file=final_path,
                workdir=workdir,
            )
        )
    return pairs


def run_aimd_presampling(atoms: Atoms, config: WorkflowConfig, state_label: str) -> list[Atoms]:
    atoms = atoms.copy()
    atoms.calc = calculator
    atoms.set_constraint(fixed_bottom_layer(atoms, z_max=config.relax.fix_z_max))

    MaxwellBoltzmannDistribution(atoms, temperature_K=config.aimd.temperature_K)
    dyn = NVTBerendsen(
        atoms,
        timestep=config.aimd.timestep_fs * fs,
        temperature_K=config.aimd.temperature_K,
        taut=config.aimd.tau_t,
    )

    sampled_frames: list[Atoms] = []
    sampled_rows: list[dict[str, object]] = []
    step_counter = {"step": 0}
    sampled_dir = Path(f"{config.structure.neb_workdir}_aimd_samples") / state_label
    os.makedirs(sampled_dir, exist_ok=True)

    def sample_frame():
        step_counter["step"] += 1
        if step_counter["step"] % config.aimd.sample_interval != 0:
            return
        frame = atoms.copy()
        frame.calc = calculator
        energy = frame.get_potential_energy()
        sampled_frames.append(frame)
        filename = f"step_{step_counter['step']:06d}.cif"
        write(str(sampled_dir / filename), frame)
        sampled_rows.append(
            {
                "step": step_counter["step"],
                "energy_eV": energy,
                "file": filename,
            }
        )

    dyn.attach(sample_frame, interval=1)
    print(f"Running AIMD presampling for {state_label} at {config.aimd.temperature_K} K")
    dyn.run(config.aimd.steps)
    if sampled_rows:
        pd.DataFrame(sampled_rows).to_csv(sampled_dir / "energies.csv", index=False)
    return sampled_frames


def coarse_relax_state(atoms: Atoms, config: WorkflowConfig, state_label: str) -> Atoms:
    atoms = atoms.copy()
    atoms.calc = calculator
    atoms.set_constraint(fixed_bottom_layer(atoms, z_max=config.relax.fix_z_max))
    print(f"Running coarse relaxation for {state_label}")
    BFGS(atoms, logfile=None).run(fmax=config.relax.coarse_relax_fmax)
    coarse_dir = Path(f"{config.structure.neb_workdir}_coarse_relaxed_states")
    os.makedirs(coarse_dir, exist_ok=True)
    write(str(coarse_dir / f"{state_label}.cif"), atoms)
    return atoms


def select_lowest_energy_sample(sampled_frames: Sequence[Atoms], config: WorkflowConfig, state_label: str) -> Atoms:
    if not sampled_frames:
        raise ValueError(f"No AIMD frames were sampled for {state_label}.")

    best_atoms: Optional[Atoms] = None
    best_energy: Optional[float] = None
    optimized_rows: list[dict[str, object]] = []
    optimized_dir = Path(f"{config.structure.neb_workdir}_optimized_samples") / state_label
    os.makedirs(optimized_dir, exist_ok=True)

    for idx, frame in enumerate(sampled_frames):
        candidate = frame.copy()
        candidate.calc = calculator
        candidate.set_constraint(fixed_bottom_layer(candidate, z_max=config.relax.fix_z_max))
        if config.aimd.relax_sampled_frames:
            BFGS(candidate, logfile=None).run(fmax=config.aimd.sampled_frame_fmax)
        energy = candidate.get_potential_energy()
        filename = f"sample_{idx:04d}.cif"
        write(str(optimized_dir / filename), candidate)
        source_step = (idx + 1) * config.aimd.sample_interval
        optimized_rows.append(
            {
                "sample_index": idx,
                "source_step": source_step,
                "energy_eV": energy,
                "file": filename,
            }
        )
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_atoms = candidate.copy()

    assert best_atoms is not None
    if optimized_rows:
        pd.DataFrame(optimized_rows).to_csv(optimized_dir / "energies.csv", index=False)
    print(f"Selected lowest-energy sampled structure for {state_label}: {best_energy} eV")
    return best_atoms


def relax_state_structures(config: WorkflowConfig) -> list[tuple[Path, Atoms]]:
    relaxed_dir = Path(f"{config.structure.neb_workdir}_relaxed_states")
    os.makedirs(relaxed_dir, exist_ok=True)

    relaxed_states: list[tuple[Path, Atoms]] = []
    for state_path in config.state_paths:
        state_label = Path(state_path).stem
        atoms = load_structure(state_path)
        if config.aimd.use_aimd_presampling:
            if config.relax.do_coarse_relax_before_aimd:
                atoms = coarse_relax_state(atoms, config, state_label)
            sampled_frames = run_aimd_presampling(atoms, config, state_label)
            atoms = select_lowest_energy_sample(sampled_frames, config, state_label)
        elif config.relax.relax_is_fs:
            relax_structure(atoms, config, fmax=config.relax.is_fmax, fix_bottom=True)
        else:
            atoms.calc = calculator
            atoms.set_constraint(fixed_bottom_layer(atoms, z_max=config.relax.fix_z_max))

        output_path = relaxed_dir / state_path.name
        write(str(output_path), atoms)
        print(f"Saved relaxed state: {output_path}")
        relaxed_states.append((output_path, atoms))

    return relaxed_states


def run_neb(filepath: str, config: WorkflowConfig):
    is_atoms = read(f"{filepath}/IS.cif")
    fs_atoms = read(f"{filepath}/FS.cif")

    is_atoms.set_constraint(fixed_bottom_layer(is_atoms, z_max=config.relax.fix_z_max))
    is_atoms.calc = calculator
    BFGS(is_atoms, logfile=None).run(fmax=config.relax.fs_fmax)
    print(f"IS {is_atoms.get_potential_energy()} eV")

    fs_atoms.set_constraint(fixed_bottom_layer(fs_atoms, z_max=config.relax.fix_z_max))
    fs_atoms.calc = calculator
    BFGS(fs_atoms, logfile=None).run(fmax=config.relax.fs_fmax)
    print(f"FS {fs_atoms.get_potential_energy()} eV")

    configs = [is_atoms.copy() for _ in range(config.neb.beads - 1)] + [fs_atoms.copy()]
    for image in configs:
        image.calc = build_calculator(calc_mode=config.calc_mode)

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


def render_neb_images(filepath: str, configs: Sequence[Atoms]) -> None:
    os.makedirs(f"{filepath}/pov_NEB", exist_ok=True)
    os.makedirs(f"{filepath}/png_NEB", exist_ok=True)
    for i, atoms in enumerate(configs):
        write(f"{filepath}/pov_NEB/NEB_{i:03}.pov", atoms.copy(), rotation="-60x, 30y, 15z")
        write(f"{filepath}/png_NEB/NEB_{i:03}.png", atoms.copy(), rotation="-60x, 30y, 15z")

    imgs = []
    for image_path in sorted(glob.glob(f"{filepath}/png_NEB/*.png")):
        image = Image.open(image_path)
        image.load()
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        imgs.append(bg)

    if imgs:
        imgs[0].save(
            f"{filepath}/gif_NEB.gif",
            save_all=True,
            append_images=imgs[1:],
            optimize=False,
            duration=100,
            loop=0,
        )


def calc_max_force(atoms: Atoms) -> float:
    return ((atoms.get_forces() ** 2).sum(axis=1).max()) ** 0.5


def get_segment_ts_index(config: WorkflowConfig, segment_index: int) -> int:
    if segment_index < len(config.neb.segment_ts_indices):
        return config.neb.segment_ts_indices[segment_index]
    return config.neb.ts_index


def analyze_neb(filepath: str, config: WorkflowConfig, ts_index: int) -> tuple[list[Atoms], list[float], list[float]]:
    configs = read(f"{filepath}/NEB_images.xyz", index=":")
    for image in configs:
        image.calc = build_calculator(calc_mode=config.calc_mode)

    energies = [config.get_total_energy() for config in configs]
    plt.plot(range(len(energies)), energies)
    plt.xlabel("replica")
    plt.ylabel("energy [eV]")
    plt.xticks(np.arange(0, len(energies), 2))
    plt.grid(True)
    plt.show()

    mforces = [calc_max_force(config) for config in configs]
    plt.plot(range(len(mforces)), mforces)
    plt.xlabel("replica")
    plt.ylabel("max force [eV]")
    plt.xticks(np.arange(0, len(mforces), 2))
    plt.grid(True)
    plt.show()

    act_energy = energies[ts_index] - energies[0]
    delta_energy = energies[ts_index] - energies[-1]
    print(f"actE {act_energy} eV, deltaE {delta_energy} eV")
    return configs, energies, mforces


def optimize_ts(filepath: str, configs: Sequence[Atoms], config: WorkflowConfig, ts_index: int) -> tuple[Atoms, pd.DataFrame]:
    ts = configs[ts_index].copy()
    ts.set_constraint(fixed_bottom_layer(ts, z_max=config.relax.fix_z_max))
    z_pos = plot_z_positions(ts)
    ts.calc = calculator
    tsopt = Sella(ts)
    tsopt.run(fmax=config.relax.ts_fmax)
    print(ts.get_potential_energy(), ts.get_forces().max())
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

    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=ts.get_potential_energy(),
        atoms=ts,
        geometry=config.vib.geometry,
        symmetrynumber=config.vib.symmetrynumber,
        spin=config.vib.spin,
        natoms=len(vibatoms),
    )
    gibbs_energy = thermo.get_gibbs_energy(temperature=config.vib.temperature, pressure=config.vib.pressure)
    print(f"Gibbs energy: {gibbs_energy} eV")

    vib.summary()
    vib.summary(log=f"{filepath}/vib_summary.txt")
    vib.write_mode(n=config.vib.mode_index, kT=config.vib.mode_kT, nimages=config.vib.mode_images)
    vib.clean()

    vib_traj = Trajectory(f"{vibpath}.{config.vib.mode_index}.traj")
    write(f"{filepath}/vib_traj.xyz", vib_traj)
    return vib, read(f"{filepath}/vib_traj.xyz", index=":")


def render_vibration_images(filepath: str, vib_traj: Sequence[Atoms]) -> None:
    os.makedirs(f"{filepath}/pov_VIB", exist_ok=True)
    os.makedirs(f"{filepath}/png_VIB", exist_ok=True)
    for i, atoms in enumerate(vib_traj):
        write(f"{filepath}/pov_VIB/VIB_{i:03}.pov", atoms.copy(), rotation="-60x, 30y, 15z")
        write(f"{filepath}/png_VIB/VIB_{i:03}.png", atoms.copy(), rotation="-60x, 30y, 15z")

    vib_energies = []
    for atoms in vib_traj:
        atoms.calc = calculator
        vib_energies.append(atoms.get_potential_energy())
    plt.plot(range(len(vib_energies)), vib_energies)
    plt.grid(True)
    plt.show()


def pseudo_irc(ts: Atoms, vib_traj: Sequence[Atoms], config: WorkflowConfig):
    constraint = fixed_bottom_layer(ts, z_max=config.relax.fix_z_max)

    irc_is = vib_traj[14].copy()
    irc_is.calc = calculator
    irc_is.set_constraint(constraint)
    opt = BFGS(irc_is, logfile=None, maxstep=0.5)
    opt.run(fmax=config.relax.irc_fmax, steps=config.relax.irc_bfgs_steps)
    print("IS_BFGS_done")
    MinimaHopping(irc_is, T0=0, fmax=config.relax.irc_fmax)(totalsteps=10)
    print("IS_MH_done")

    irc_fs = vib_traj[16].copy()
    irc_fs.calc = calculator
    irc_fs.set_constraint(constraint)
    opt = BFGS(irc_fs, logfile=None, maxstep=0.5)
    opt.run(fmax=config.relax.irc_fmax, steps=config.relax.irc_bfgs_steps)
    print("FS_BFGS_done")

    return irc_is, irc_fs


def compare_neb_and_irc(configs: Sequence[Atoms], ts_index: int, ts: Atoms, irc_is: Atoms, irc_fs: Atoms):
    plt.plot(
        [0, 1, 2],
        [configs[0].get_potential_energy(), configs[ts_index].get_potential_energy(), configs[-1].get_potential_energy()],
        label="NEB",
    )
    plt.plot(
        [0, 1, 2],
        [irc_is.get_potential_energy(), ts.get_potential_energy(), irc_fs.get_potential_energy()],
        label="TSopt+IRC",
    )
    plt.legend()
    plt.grid(True)
    plt.show()
    print(ts.get_potential_energy() - irc_is.get_potential_energy())
    print(ts.get_potential_energy() - irc_fs.get_potential_energy())


def run_single_segment(filepath: str, config: WorkflowConfig, segment_index: int) -> None:
    ts_index = get_segment_ts_index(config, segment_index)
    run_neb(filepath, config)
    configs, _, _ = analyze_neb(filepath, config, ts_index)
    ts, z_pos = optimize_ts(filepath, configs, config, ts_index)
    _, vib_traj = run_vibrational_analysis(filepath, ts, z_pos, config)
    irc_is, irc_fs = pseudo_irc(ts, vib_traj, config)
    compare_neb_and_irc(configs, ts_index, ts, irc_is, irc_fs)
    write(f"{filepath}/IS_IRC.xyz", irc_is)
    write(f"{filepath}/FS_IRC.xyz", irc_fs)


def build_runtime_config_from_relaxed_states(
    config: WorkflowConfig,
    relaxed_states: Sequence[tuple[Path, Atoms]],
) -> WorkflowConfig:
    relaxed_state_files = [path for path, _ in relaxed_states]
    return WorkflowConfig(
        calc_mode=config.calc_mode,
        compute_adsorption_energy=config.compute_adsorption_energy,
        only_relax_states=config.only_relax_states,
        run_neb=config.run_neb,
        adsorption_pairs=config.adsorption_pairs,
        structure=StructureConfig(
            input_dir=Path("."),
            molecule_file=config.structure.molecule_file,
            is_file=config.structure.is_file,
            fs_file=config.structure.fs_file,
            state_files=[str(path) for path in relaxed_state_files],
            output_dir=config.structure.output_dir,
            structures_dir=config.structure.structures_dir,
            adsorbate_dir=config.structure.adsorbate_dir,
            neb_workdir=config.structure.neb_workdir,
        ),
        relax=config.relax,
        neb=config.neb,
        vib=config.vib,
        aimd=config.aimd,
    )


def run_adsorption_block(
    config: WorkflowConfig = CONFIG,
    relaxed_state_dir: Optional[Path] = None,
) -> list[tuple[str, float]]:
    if not config.compute_adsorption_energy:
        print("compute_adsorption_energy=False, skipping adsorption block.")
        return []
    return run_adsorption_energy_pairs(config, relaxed_state_dir=relaxed_state_dir)


def prepare_screened_states(config: WorkflowConfig = CONFIG) -> tuple[list[tuple[Path, Atoms]], Path]:
    relaxed_states = relax_state_structures(config)
    relaxed_state_dir = Path(f"{config.structure.neb_workdir}_relaxed_states")
    return relaxed_states, relaxed_state_dir


def run_neb_block(
    config: WorkflowConfig = CONFIG,
    relaxed_states: Optional[Sequence[tuple[Path, Atoms]]] = None,
) -> None:
    if not config.run_neb:
        print("run_neb=False, skipping NEB block.")
        return

    if relaxed_states is None:
        relaxed_states, _ = prepare_screened_states(config)

    if config.only_relax_states:
        return

    runtime_config = build_runtime_config_from_relaxed_states(config, relaxed_states)
    segment_pairs = prepare_neb_pairs_from_states(runtime_config)
    for segment_index, (filepath, _, _) in enumerate(segment_pairs):
        print(f"Running NEB segment: {filepath}")
        run_single_segment(filepath, runtime_config, segment_index)


def main() -> None:
    relaxed_states, relaxed_state_dir = prepare_screened_states(CONFIG)
    run_adsorption_block(CONFIG, relaxed_state_dir=relaxed_state_dir)
    run_neb_block(CONFIG, relaxed_states=relaxed_states)


if __name__ == "__main__":
    main()
