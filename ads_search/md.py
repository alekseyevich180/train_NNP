import argparse
import time
import weakref
from pathlib import Path
from typing import IO, Any, Union

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import PropertyNotImplementedError
from ase.eos import EquationOfState
from ase.io import read
from ase.md import MDLogger
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.parallel import world
from ase.utils import IOContext
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import torch, warnings

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 on A100/H100


class MDLogger(IOContext):
    """Class for logging molecular dynamics simulations."""

    def __init__(
        self,
        dyn: Any,  # not fully annotated so far to avoid a circular import
        atoms: Atoms,
        logfile: Union[IO, str],
        stress: bool = False,
        mode: str = "a",
        comm=world,
    ):
        """
        Args:
            dyn (Any): The dynamics.  Only a weak reference is kept.
            atoms (Atoms): The atoms.
            logfile (Union[IO, str]): File name or open file, "-" meaning standard output.
            stress (bool, optional): Include stress in log.
            mode (str, optional): How the file is opened if logfile is a filename.
        """
        self.dyn = weakref.proxy(dyn) if hasattr(dyn, "get_time") else None
        self.atoms = atoms
        global_natoms = atoms.get_global_number_of_atoms()
        self.logfile = self.openfile(file=logfile, mode=mode, comm=comm)
        self.stress = stress
        self.hdr = "%-9s %-9s %7s %12s %12s %12s  %12s" % (
            "Step",
            "Time[ps]",
            "T[K]",
            "Epot[eV]",
            "Ekin[eV]",
            "Etot[eV]",
            "Density[g/cm3]",
        )
        # Choose a sensible number of decimals
        if global_natoms <= 100:
            digits = 4
        elif global_natoms <= 1000:
            digits = 3
        elif global_natoms <= 10000:
            digits = 2
        else:
            digits = 1
        self.fmt = "%-10d %-10.4f %6.1f" + 4 * ("%%12.%df " % (digits))
        if self.stress:
            self.hdr += "   ----------------------- stress [GPa] ------------------------"
            self.fmt += 6 * " %10.3f"
        self.fmt += "\n"
        self.logfile.write(self.hdr + "\n")

    def __del__(self):
        self.close()

    def __call__(self):
        if self.dyn:
            t = self.dyn.get_time() / (1000 * units.fs)
            temp = self.atoms.get_temperature()
            epot = self.atoms.get_potential_energy()
            ekin = self.atoms.get_kinetic_energy()
            density = (sum(self.atoms.get_masses()) / units.mol) / (self.atoms.get_volume() * 1e-24)
            dat = (self.dyn.nsteps, t, temp, epot, ekin, epot + ekin, density)
            if self.stress:
                dat += tuple(self.atoms.get_stress(include_ideal_gas=True) / units.GPa)
            self.logfile.write(self.fmt % dat)
            self.logfile.flush()


def shape_upper_triangular_cell(atoms: Atoms) -> Atoms:
    """Transform to upper-triangular cell.

    Args:
        atoms (Atoms): Atoms objects

    Returns:
        atoms (Atoms): Atoms objects whose cell is shaped to upper-triangular cell
    """
    if not NPT._isuppertriangular(atoms.get_cell()):
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        angles = np.radians((alpha, beta, gamma))
        sin_a, sin_b, _sin_g = np.sin(angles)
        cos_a, cos_b, cos_g = np.cos(angles)
        cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
        cos_p = np.clip(cos_p, -1, 1)
        sin_p = (1 - cos_p**2) ** 0.5

        new_basis = [
            (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
            (0, b * sin_a, b * cos_a),
            (0, 0, c),
        ]
        atoms.set_cell(new_basis, scale_atoms=True)
    return atoms


def calculate_eos(atoms: Atoms, linspace_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate energy and volume for given atoms.

    Args:
        atoms (Atoms): ASE atoms object
        linspace_step (int, optional): define interval of volumes. Defaults to 20.

    Returns:
        volumes (np.ndarray): volumes of system (range: 0.95 to 1.05)
        energies (np.ndarray): inferenced energies
    """
    volumes = []
    energies = []
    base_cell = atoms.get_cell()
    # DFT(scf) or NNP inference with different sizes of cell
    for x in np.linspace(0.95, 1.05, linspace_step):
        atoms.set_cell(base_cell * x, scale_atoms=True)
        volume = atoms.get_volume()
        energy = atoms.get_potential_energy() / len(atoms)
        volumes.append(volume)
        energies.append(energy)

    return np.array(volumes), np.array(energies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--package", type=str, default=None)
    parser.add_argument("--out_traj_path", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--run_steps", type=int, default=200000)
    parser.add_argument("--traj_interval", type=int, default=100)
    parser.add_argument("--ensemble", type=str, default="nvt", choices=["nvt", "npt"])
    parser.add_argument("--taut", type=int, default=100)
    parser.add_argument("--pressure", type=float, default=1.0)
    parser.add_argument("--taup", type=int, default=1000)
    parser.add_argument("--ensemble_model_paths", type=Path, nargs="*")
    parser.add_argument("--include_d3", action="store_true")
    parser.add_argument("--uma_model", type=str, default="uma-s-1p1",
                    help="UMA model name, e.g., uma-s-1p1 / uma-m-1p1")
    parser.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"], help="inference device")

    args = parser.parse_args()

    print(args)
    args.out_traj_path.parent.mkdir(exist_ok=True, parents=True)
    atoms = read(args.atoms_path)
    atoms = shape_upper_triangular_cell(atoms)

    ### SET CALCULATOR ###
    
    predictor = pretrained_mlip.get_predict_unit(args.uma_model, device=args.device)
    calc = FAIRChemCalculator(predictor, task_name="omat")

    ######################

    if args.include_d3:
        ### DFTD3 implemented in ASE and https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/
        from ase.calculators.dftd3 import DFTD3

        calc = DFTD3(dft=calc)

        ### TorchDFTD3 ###
        # from ase.calculators.mixing import SumCalculator
        # from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        # d3 = TorchDFTD3Calculator(atoms=atoms, device="cpu")
        # calc = SumCalculator([calc, d3])
    atoms.calc = calc
    
    # set momenta
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, force_temp=True)
    Stationary(atoms)

    if args.ensemble == "nvt":
        dyn = NPT(
            atoms,
            args.timestep * units.fs,
            temperature_K=args.temperature,
            externalstress=0,
            ttime=args.taut * units.fs,
            pfactor=None,
            trajectory=str(args.out_traj_path),
            loginterval=args.traj_interval,
        )
        logger = MDLogger(dyn, atoms, logfile="-", mode="w", stress=False)
    elif args.ensemble == "npt":
        if args.ensemble_model_paths:
            calc.calculate_model_devi = False
        # Check whether calc can compute stress property
        _atoms = atoms.copy()
        _atoms.calc = calc
        try:
            _atoms.get_stress()
        except PropertyNotImplementedError:
            print("Calculator cannot compute stress property")
            return

        # compute bulk modulus
        print("Computing bulk modulus...")
        v, e = calculate_eos(_atoms, 100)
        eos = EquationOfState(v, e, eos="murnaghan")
        _, _, B = eos.fit()
        bulk_modulus_GPa = B / units.kJ * 1e24
        print(f"Bulk Modulus: {bulk_modulus_GPa} GPa")

        if args.ensemble_model_paths:
            calc.calculate_model_devi = True
        dyn = NPT(
            atoms,
            args.timestep * units.fs,
            temperature_K=args.temperature,
            externalstress=args.pressure * units.bar,
            ttime=args.taut * units.fs,
            pfactor=(args.taup * units.fs) ** 2 * bulk_modulus_GPa * units.GPa,
            mask=np.identity(3),
            trajectory=str(args.out_traj_path),
            loginterval=args.traj_interval,
        )
        logger = MDLogger(dyn, atoms, logfile="-", mode="a", stress=True)
    else:
        print(f"Unsupported ensemble: {args.ensemble}")
        return

    dyn.attach(logger, interval=args.traj_interval)

    # run md simulation
    time_start = time.perf_counter()
    print("Starting MD simulation...")
    dyn.run(args.run_steps)
    time_end = time.perf_counter()
    print(f"MD simulation finished in {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
