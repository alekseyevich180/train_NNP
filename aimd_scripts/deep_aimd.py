import os
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize import LBFGS
import pfp_api_client
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator


CONFIG={
"system":{
"input_file":"acid.cif",
"fixed_z_lower_bound":4.0,
"fixed_z_upper_bound":9.0,
"surface_relax_depth":12.0,
"output_root":"acid_AIMD_dataset"
},
"md_control":{
"initial_temp":280,
"final_temp":1080,
"ramp_interval":100,
"ramp_steps":20000,
"stab_steps":10000,
"prod_steps":2000000,
"timestep":0.5,
"tau_t":100.0
},
"output":{
"save_interval":50,
"deepmd_set_size":1000,
"deepmd_dir":"deepmd_dataset",
"cif_dir":"cif_frames",
"cif_set_interval":100000
}
}


# =========================================================
# Surface relaxation
# 只优化表面 ZnO
# =========================================================

def relax_surface(atoms):

    print("Starting surface relaxation...")

    fixed_z_lower = CONFIG["system"]["fixed_z_lower_bound"]
    fixed_z_upper = CONFIG["system"]["fixed_z_upper_bound"]
    surface_depth = CONFIG["system"]["surface_relax_depth"]

    max_z = max(atoms.positions[:,2])
    surface_z = max_z - surface_depth

    freeze=[]

    for atom in atoms:

        # 冻结水和有机物
        if atom.symbol not in ["Zn","O"]:
            freeze.append(atom.index)

        # 冻结固定 z 区间内的 ZnO
        elif fixed_z_lower <= atom.position[2] <= fixed_z_upper:
            freeze.append(atom.index)

        # 冻结深层 ZnO（只优化最表面几层）
        elif atom.position[2] < surface_z:
            freeze.append(atom.index)

    atoms.set_constraint(FixAtoms(indices=freeze))

    opt=LBFGS(atoms,logfile="surface_relax.log")
    opt.run(fmax=0.05)

    atoms.set_constraint()

    print("Surface relaxation finished")


# =========================================================
# DeepMD dataset writer
# =========================================================

class DeepMDWriter:

    def __init__(self,atoms,root,set_size):

        self.root=root
        self.set_size=set_size

        os.makedirs(root,exist_ok=True)

        self.coord=[]
        self.force=[]
        self.energy=[]
        self.box=[]

        self.frame=0
        self.set_id=0

        self.write_type_files(atoms)

    def write_type_files(self,atoms):

        symbols=atoms.get_chemical_symbols()
        uniq=sorted(set(symbols))

        type_map={s:i for i,s in enumerate(uniq)}
        type_list=[type_map[s] for s in symbols]

        np.savetxt(os.path.join(self.root,"type.raw"),np.array(type_list),fmt="%d")

        with open(os.path.join(self.root,"type_map.raw"),"w") as f:
            for s in uniq:
                f.write(s+"\n")

    def add_frame(self,atoms):

        self.coord.append(atoms.get_positions().reshape(-1))
        self.force.append(atoms.get_forces().reshape(-1))
        self.energy.append(atoms.get_potential_energy())
        self.box.append(atoms.get_cell().array.reshape(-1))

        self.frame+=1

        if self.frame>=self.set_size:

            self.write_set()
            self.reset()

    def reset(self):

        self.coord=[]
        self.force=[]
        self.energy=[]
        self.box=[]
        self.frame=0

    def write_set(self):

        set_dir=os.path.join(self.root,f"set.{self.set_id:03d}")
        os.makedirs(set_dir,exist_ok=True)

        np.save(os.path.join(set_dir,"coord.npy"),np.array(self.coord))
        np.save(os.path.join(set_dir,"force.npy"),np.array(self.force))
        np.save(os.path.join(set_dir,"energy.npy"),np.array(self.energy))
        np.save(os.path.join(set_dir,"box.npy"),np.array(self.box))

        print(f"write {set_dir}")

        self.set_id+=1


# =========================================================
# 主程序
# =========================================================

def run():

    root=CONFIG["system"]["output_root"]

    deepmd_root=os.path.join(root,CONFIG["output"]["deepmd_dir"])
    cif_root=os.path.join(root,CONFIG["output"]["cif_dir"])

    os.makedirs(cif_root,exist_ok=True)

    atoms=read(CONFIG["system"]["input_file"])

    estimator=Estimator(calc_mode=EstimatorCalcMode.PBE_U_PLUS_D3)
    calculator=ASECalculator(estimator)

    atoms.calc=calculator
    atoms.pbc=True


    # -------------------------------------------------
    # 1 surface relaxation
    # -------------------------------------------------

    relax_surface(atoms)


    # -------------------------------------------------
    # 2 AIMD 阶段约束
    # 只固定底层 ZnO
    # -------------------------------------------------

    fixed_z_lower = CONFIG["system"]["fixed_z_lower_bound"]
    fixed_z_upper = CONFIG["system"]["fixed_z_upper_bound"]
    fixed=[
        a.index
        for a in atoms
        if fixed_z_lower <= a.position[2] <= fixed_z_upper
    ]
    atoms.set_constraint(FixAtoms(indices=fixed))


    writer=DeepMDWriter(atoms,deepmd_root,CONFIG["output"]["deepmd_set_size"])


    ctrl=CONFIG["md_control"]

    curr_t=ctrl["initial_temp"]
    final_t=ctrl["final_temp"]


    MaxwellBoltzmannDistribution(atoms,temperature_K=curr_t)


    dyn=NVTBerendsen(
        atoms,
        timestep=ctrl["timestep"]*fs,
        temperature_K=curr_t,
        taut=ctrl["tau_t"]
    )


    step_counter={"step":0}


    def save_frame():

        step_counter["step"]+=1

        if step_counter["step"]%CONFIG["output"]["save_interval"]!=0:
            return


        writer.add_frame(atoms)


        set_id=step_counter["step"]//CONFIG["output"]["cif_set_interval"]

        cif_set_dir=os.path.join(cif_root,f"set_{set_id:03d}")
        os.makedirs(cif_set_dir,exist_ok=True)

        cif_file=os.path.join(
            cif_set_dir,
            f"step_{step_counter['step']:08d}.cif"
        )

        write(cif_file,atoms)


    dyn.attach(save_frame,interval=1)


    print(f"Initial equilibration at {curr_t} K")
    dyn.run(10000)


    while curr_t<final_t:

        curr_t+=ctrl["ramp_interval"]

        if curr_t>final_t:
            curr_t=final_t

        dyn.set_temperature(temperature_K=curr_t)

        print(f"Ramp to {curr_t} K")

        dyn.run(ctrl["ramp_steps"])
        dyn.run(ctrl["stab_steps"])


    print(f"Production run at {final_t} K")

    dyn.run(ctrl["prod_steps"])


if __name__=="__main__":
    run()
