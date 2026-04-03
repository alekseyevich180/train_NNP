import io
import os
import pandas as pd
import tempfile

from ase import Atoms
from ase.build import bulk, fcc111, molecule, add_adsorbate
from ase.constraints import ExpCellFilter, StrainFilter
from ase.io import write, read
from ase.io.jsonio import write_json, read_json
from ase.optimize import LBFGS, FIRE
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import optuna
from ase.visualize import view


import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode

from pfcc_extras.visualize.view import view_ngl
from pfcc_extras.visualize.ase import view_ase_atoms

print(f"pfp_api_client: {pfp_api_client.__version__}")

# estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL, model_version="latest")
estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0, model_version="v3.0.0")
calculator = ASECalculator(estimator)

def get_opt_energy(atoms, fmax=0.001, opt_mode: str = "normal"):    
    atoms.set_calculator(calculator)
    if opt_mode == "scale":
        opt1 = LBFGS(StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None)
    elif opt_mode == "all":
        opt1 = LBFGS(ExpCellFilter(atoms), logfile=None)
    else:
        opt1 = LBFGS(atoms, logfile=None)
    opt1.run(fmax=fmax)
    return atoms.get_total_energy()

# y = x^2 (0 <= x <= 1)を最小化する例
def objective(trial):
    x = trial.suggest_float("x", 0, 1)
    return x ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=30)
optuna.visualization.plot_optimization_history(study)

#build structure
def create_slab():
    a = np.mean(np.diag(bulk_atoms.cell))
    slab =  fcc111("Pt", a=a, size=(4, 4, 4), vacuum=40.0, periodic=True)
    slab.calc = calculator
    E_slab = get_opt_energy(slab, fmax=1e-4, opt_mode="normal")
    return slab, E_slab 

#slab, E_slab = create_slab()
#view_ngl(slab, representations=["ball+stick"])

def create_mol():
    mol = molecule("CO")
    mol.calc = calculator
    E_mol = get_opt_energy(mol, fmax=1e-4)
    return mol, E_mol

#mol, E_mol = create_mol()
#view_ngl(mol, representations=["ball+stick"])

def already_slab():
    slab = read("relaxed.cif")
    slab.calc = calculator
    E_slab = get_opt_energy(slab, fmax=1e-4, opt_mode="normal")
    return slab, E_slab 
slab, E_slab = already_slab()
view_ngl(slab, representations=["ball+stick"])

def already_mol(filename):
    mol = read(filename)
    mol.calc = calculator  # 你已有的 calculator 定义
    E_mol = get_opt_energy(mol, fmax=1e-4)
    return mol, E_mol
#mol, E_mol = already_mol()
#view_ngl(mol, representations=["ball+stick"])

#mol_list = ["2-ketone.cif", "3-ketone.cif", "4-ketone.cif", "5-ketone.cif"]

def get_all_molecules():
    molecules = []
    for fname in ["S.vasp","G.vasp","H.vasp"]:
        mol, E_mol = already_mol(fname)
        molecules.append((mol, E_mol, fname))
    return molecules

def adjust_adsorption_height(slab, mol, hight=2.0):
    slab_positions = slab.get_positions()
    mol_positions = mol.get_positions()

    max_slab_z = np.max(slab_positions[:, 2])

    min_mol_z = np.min(mol_positions[:, 2])

    shift_z = (max_slab_z + hight) - min_mol_z
    mol.positions[:, 2] += shift_z

    return mol

#search ads for big mol
def atoms_to_json(atoms):
    f = io.StringIO()
    write(f, atoms, format="json")
    return f.getvalue()


def json_to_atoms(atoms_str):
    return read(io.StringIO(atoms_str), format="json")

for mol, E_mol, name in get_all_molecules():
    print(f"🔍 Molecule: {name}")
    mol_json_str = atoms_to_json(mol)
    mol2 = json_to_atoms(mol_json_str)
    
    print(f"{mol_json_str=}")
    view_ngl(mol2, representations=["ball+stick"])    

#mol_json_str = atoms_to_json(mol)
#mol2 = json_to_atoms(mol_json_str)

#print(f"{mol_json_str=}")
#view_ngl(mol2, representations=["ball+stick"])

def objective(trial):
    slab = json_to_atoms(trial.study.user_attrs["slab"])
    E_slab = trial.study.user_attrs["E_slab"]
    mol = json_to_atoms(trial.study.user_attrs["mol"])
    E_mol = trial.study.user_attrs["E_mol"]

    phi = 180. * trial.suggest_float("phi", -1, 1)
    theta = np.arccos(trial.suggest_float("theta", -1, 1)) * 180. / np.pi
    psi = 180. * trial.suggest_float("psi", -1, 1)
    x_pos = trial.suggest_float("x_pos", 0, 0.5)
    y_pos = trial.suggest_float("y_pos", 0, 0.5)
    z_hig = trial.suggest_float("z_hig", 2.0, 6.0)

    mol.euler_rotate(phi=phi, theta=theta, psi=psi)

    xy_position = np.matmul([x_pos, y_pos, 0], slab.cell)[:2]
    mol.translate([*xy_position, 0.0])

    max_slab_z = np.max(slab.get_positions()[:, 2])
    min_mol_z = np.min(mol.get_positions()[:, 2])
    shift_z = (max_slab_z + z_hig) - min_mol_z
    mol.translate([0, 0, shift_z])

    combined = slab + mol
    combined.calc = calculator

    E_slab_mol = get_opt_energy(combined, fmax=1e-3)
    trial.set_user_attr("structure", atoms_to_json(combined))

    return E_slab_mol - E_slab - E_mol

for mol, E_mol, name in get_all_molecules():
    print(f"\n🔍 Starting optimization for {name}...\n")
    slab, E_slab = already_slab()
    view(mol, viewer="ngl")

    study = optuna.create_study()
    study.set_user_attr("slab", atoms_to_json(slab))
    study.set_user_attr("E_slab", E_slab)
    study.set_user_attr("mol", atoms_to_json(mol))
    study.set_user_attr("E_mol", E_mol)

    study.optimize(objective, n_trials=400)

    print(f"    Best trial for {name} is #{study.best_trial.number}")
    print(f"    Adsorption energy: {study.best_value:.6f} eV")
    print("    Adsorption position:")
    for key in ["phi", "theta", "psi", "x_pos", "y_pos", "z_hig"]:
        print(f"        {key}: {study.best_params[key]}")

    output_dir = os.path.join("output_ligin", name)
    os.makedirs(output_dir, exist_ok=True)

    # Save optimization plots
    optuna.visualization.plot_optimization_history(study).write_html(
        os.path.join(output_dir, "optimization_history.html"))
    optuna.visualization.plot_slice(study).write_html(
        os.path.join(output_dir, "optimization_slice.html"))

    # Save best trial structure
    best_slab = json_to_atoms(study.best_trial.user_attrs["structure"])
    view_ngl(best_slab, representations=["ball+stick"])
    write(os.path.join(output_dir, "best_trial.cif"), best_slab)

    # Save all trials' structures and images
    trial_data = []
    n_trials = len(study.trials)
    fig_rows = (n_trials // 10) + 1
    fig, axes = plt.subplots(fig_rows, 10, figsize=(20, 2 * fig_rows))

    if fig_rows == 1:
        axes = [axes]  # flatten if only one row

    for trial in study.trials:
        slab = json_to_atoms(trial.user_attrs["structure"])

        # Save structure file
        trial_cif = os.path.join(output_dir, f"{trial.number}.cif")
        write(trial_cif, slab)

        # Save image
        img_path = os.path.join(output_dir, f"{trial.number}.png")
        write(img_path, slab, rotation="0x,0y,90z")
        ax = axes[trial.number // 10][trial.number % 10]
        ax.imshow(mpimg.imread(img_path))
        ax.set_axis_off()
        ax.set_title(trial.number)

        # Collect trial data
        trial_data.append({
            "trial": trial.number,
            "energy": trial.value,
            **trial.params
        })

    # Save energy data CSV
    df = pd.DataFrame(trial_data)
    df.to_csv(os.path.join(output_dir, "data.csv"), index=False)

    # Save grid plot of all trials
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trial_grid.png"))
    plt.show()

    # Show all trial structures in viewer
    slabs = [json_to_atoms(trial.user_attrs["structure"]) for trial in study.trials]
    view_ngl(slabs, representations=["ball+stick"], replace_structure=True)