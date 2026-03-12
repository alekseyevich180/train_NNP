import os
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.units import fs, Gpa
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
import pfp_api_client
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

CONFIG={
"system":{
"input_file":"aimd_bigger.cif",
"bottom_z_threshold":9.0,
"output_root":"C9_AIMD_dataset"
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
"save_interval":100,
"deepmd_set_size":1000,
"deepmd_dir":"deepmd_dataset",
"cif_dir":"cif_frames"
}
}

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
    fixed=[a.index for a in atoms if a.position[2]<CONFIG["system"]["bottom_z_threshold"]]
    atoms.set_constraint(FixAtoms(indices=fixed))
    writer=DeepMDWriter(atoms,deepmd_root,CONFIG["output"]["deepmd_set_size"])
    ctrl=CONFIG["md_control"]
    curr_t=ctrl["initial_temp"]
    final_t=ctrl["final_temp"]
    MaxwellBoltzmannDistribution(atoms,temperature_K=curr_t)
    dyn=NVTBerendsen(atoms,timestep=ctrl["timestep"]*fs,temperature_K=curr_t,taut=ctrl["tau_t"])
    step_counter={"step":0}

    def save_frame():
        step_counter["step"]+=1
        if step_counter["step"]%CONFIG["output"]["save_interval"]!=0:
            return
        writer.add_frame(atoms)
        cif_file=os.path.join(cif_root,f"frame_{step_counter['step']:08d}.cif")
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