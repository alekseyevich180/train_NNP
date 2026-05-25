import json
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
"input_file":"ketone.cif",
"fixed_z_lower_bound":4.0,
"fixed_z_upper_bound":9.0,
"surface_relax_depth":12.0,
"output_root":"acid_AIMD_dataset",
"restart_from":None
},
"relaxation":{
"surface_fmax":0.05,
"whole_fmax":0.05
},
"md_control":{
"initial_temp":280,
"final_temp":1080,
"ramp_interval":100,
"ramp_steps":20000,
"stab_steps":10000,
"prod_steps":4000000,
"timestep":0.5,
"tau_t":100.0
},
"output":{
"save_interval":50,
"deepmd_dir":"deepmd_dataset",
"cif_dir":"cif_frames",
"restart_dir":"restart_checkpoints",
"deepmd_set_interval":100000,
"cif_set_interval":100000,
"restart_interval":200000
}
}


# =========================================================
# Relaxation helpers
# =========================================================

def get_aimd_fixed_indices(atoms):

    fixed_z_lower = CONFIG["system"]["fixed_z_lower_bound"]
    fixed_z_upper = CONFIG["system"]["fixed_z_upper_bound"]

    return [
        atom.index
        for atom in atoms
        if fixed_z_lower <= atom.position[2] <= fixed_z_upper
    ]


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
    opt.run(fmax=CONFIG["relaxation"]["surface_fmax"])

    atoms.set_constraint()

    print("Surface relaxation finished")


def relax_whole_structure(atoms):

    print("Starting whole-structure relaxation...")

    fixed = get_aimd_fixed_indices(atoms)
    atoms.set_constraint(FixAtoms(indices=fixed))

    opt=LBFGS(atoms,logfile="whole_relax.log")
    opt.run(fmax=CONFIG["relaxation"]["whole_fmax"])

    atoms.set_constraint()

    print("Whole-structure relaxation finished")


# =========================================================
# DeepMD dataset writer
# =========================================================

class DeepMDWriter:

    def __init__(self,atoms,root):

        self.root=root

        os.makedirs(root,exist_ok=True)

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

    def add_frame(self,atoms,step_id):

        set_id=step_id//CONFIG["output"]["deepmd_set_interval"]
        set_root=os.path.join(self.root,f"set_{set_id:03d}")
        set_dir=os.path.join(set_root,f"set.{step_id}")
        os.makedirs(set_dir,exist_ok=True)

        np.save(
            os.path.join(set_dir,"coord.npy"),
            np.array([atoms.get_positions().reshape(-1)])
        )
        np.save(
            os.path.join(set_dir,"force.npy"),
            np.array([atoms.get_forces().reshape(-1)])
        )
        np.save(
            os.path.join(set_dir,"energy.npy"),
            np.array([atoms.get_potential_energy()])
        )
        np.save(
            os.path.join(set_dir,"box.npy"),
            np.array([atoms.get_cell().array.reshape(-1)])
        )

        print(f"write {set_dir}")


# =========================================================
# Restart helpers
# =========================================================

def get_pressure_info(atoms):

    try:
        stress=atoms.get_stress(voigt=True)
    except Exception as exc:
        return {
            "stress_eV_A3":None,
            "pressure_GPa":None,
            "pressure_error":str(exc)
        }

    pressure_ev_a3=-float(np.mean(stress[:3]))
    return {
        "stress_eV_A3":[float(x) for x in stress],
        "pressure_GPa":pressure_ev_a3*160.21766208,
        "pressure_error":None
    }


def save_restart(atoms,restart_root,step_id,target_temperature,phase):

    checkpoint_dir=os.path.join(restart_root,f"checkpoint_{step_id:08d}")
    os.makedirs(checkpoint_dir,exist_ok=True)

    traj_file=os.path.join(checkpoint_dir,"atoms.traj")
    cif_file=os.path.join(checkpoint_dir,"atoms.cif")
    state_file=os.path.join(checkpoint_dir,"state.json")

    write(traj_file,atoms)
    write(cif_file,atoms)

    state={
        "step":int(step_id),
        "phase":phase,
        "target_temperature_K":float(target_temperature),
        "instant_temperature_K":float(atoms.get_temperature()),
        "timestep_fs":float(CONFIG["md_control"]["timestep"]),
        "tau_t_fs":float(CONFIG["md_control"]["tau_t"]),
        "input_file":CONFIG["system"]["input_file"],
        "calc_mode":"PBE_U_PLUS_D3"
    }
    state.update(get_pressure_info(atoms))

    with open(state_file,"w") as f:
        json.dump(state,f,indent=2)

    latest_file=os.path.join(restart_root,"latest_checkpoint.txt")
    with open(latest_file,"w") as f:
        f.write(checkpoint_dir+"\n")

    print(f"restart checkpoint saved: {checkpoint_dir}")


def load_restart(restart_from,calculator):

    if os.path.isdir(restart_from):
        checkpoint_dir=restart_from
        traj_file=os.path.join(checkpoint_dir,"atoms.traj")
        state_file=os.path.join(checkpoint_dir,"state.json")
    else:
        with open(restart_from) as f:
            checkpoint_dir=f.readline().strip()
        traj_file=os.path.join(checkpoint_dir,"atoms.traj")
        state_file=os.path.join(checkpoint_dir,"state.json")

    atoms=read(traj_file)
    with open(state_file) as f:
        state=json.load(f)

    atoms.calc=calculator
    atoms.pbc=True
    fixed=get_aimd_fixed_indices(atoms)
    atoms.set_constraint(FixAtoms(indices=fixed))

    print(f"Restart from {checkpoint_dir}")
    print(f"Restart step = {state['step']}, target T = {state['target_temperature_K']} K")

    return atoms,state


def make_md_segments():

    ctrl=CONFIG["md_control"]
    segments=[
        {
            "phase":"initial",
            "temperature":ctrl["initial_temp"],
            "steps":10000
        }
    ]

    curr_t=ctrl["initial_temp"]
    final_t=ctrl["final_temp"]

    while curr_t<final_t:

        curr_t+=ctrl["ramp_interval"]

        if curr_t>final_t:
            curr_t=final_t

        segments.append({
            "phase":"ramp",
            "temperature":curr_t,
            "steps":ctrl["ramp_steps"]
        })
        segments.append({
            "phase":"stabilization",
            "temperature":curr_t,
            "steps":ctrl["stab_steps"]
        })

    segments.append({
        "phase":"production",
        "temperature":final_t,
        "steps":ctrl["prod_steps"]
    })

    return segments


# =========================================================
# 主程序
# =========================================================

def run():

    root=CONFIG["system"]["output_root"]

    deepmd_root=os.path.join(root,CONFIG["output"]["deepmd_dir"])
    cif_root=os.path.join(root,CONFIG["output"]["cif_dir"])
    restart_root=os.path.join(root,CONFIG["output"]["restart_dir"])

    os.makedirs(cif_root,exist_ok=True)
    os.makedirs(restart_root,exist_ok=True)

    estimator=Estimator(calc_mode=EstimatorCalcMode.PBE_U_PLUS_D3)
    calculator=ASECalculator(estimator)

    restart_from=CONFIG["system"]["restart_from"]

    if restart_from:

        atoms,restart_state=load_restart(restart_from,calculator)
        start_step=int(restart_state["step"])

    else:

        atoms=read(CONFIG["system"]["input_file"])

        atoms.calc=calculator
        atoms.pbc=True


        # -------------------------------------------------
        # 1 surface relaxation
        # -------------------------------------------------

        relax_surface(atoms)


        # -------------------------------------------------
        # 2 whole-structure relaxation
        # 除底层固定区外整体再优化一次
        # -------------------------------------------------

        relax_whole_structure(atoms)


        # -------------------------------------------------
        # 3 AIMD 阶段约束
        # 只固定底层 ZnO
        # -------------------------------------------------

        fixed = get_aimd_fixed_indices(atoms)
        atoms.set_constraint(FixAtoms(indices=fixed))

        start_step=0


    writer=DeepMDWriter(atoms,deepmd_root)


    ctrl=CONFIG["md_control"]

    if not restart_from:
        MaxwellBoltzmannDistribution(atoms,temperature_K=ctrl["initial_temp"])


    dyn=NVTBerendsen(
        atoms,
        timestep=ctrl["timestep"]*fs,
        temperature_K=ctrl["initial_temp"],
        taut=ctrl["tau_t"]
    )


    step_counter={"step":start_step}
    run_state={
        "phase":"not_started",
        "target_temperature":ctrl["initial_temp"]
    }


    def save_frame():

        step_counter["step"]+=1

        if step_counter["step"]%CONFIG["output"]["save_interval"]==0:

            writer.add_frame(atoms,step_counter["step"])


            set_id=step_counter["step"]//CONFIG["output"]["cif_set_interval"]

            cif_set_dir=os.path.join(cif_root,f"set_{set_id:03d}")
            os.makedirs(cif_set_dir,exist_ok=True)

            cif_file=os.path.join(
                cif_set_dir,
                f"step_{step_counter['step']:08d}.cif"
            )

            write(cif_file,atoms)

        if step_counter["step"]%CONFIG["output"]["restart_interval"]==0:
            save_restart(
                atoms,
                restart_root,
                step_counter["step"],
                run_state["target_temperature"],
                run_state["phase"]
            )


    dyn.attach(save_frame,interval=1)

    completed_steps=start_step
    cumulative_steps=0

    for segment in make_md_segments():

        segment_start=cumulative_steps
        segment_end=cumulative_steps+segment["steps"]
        cumulative_steps=segment_end

        if completed_steps>=segment_end:
            continue

        remaining_steps=segment_end-max(completed_steps,segment_start)

        run_state["phase"]=segment["phase"]
        run_state["target_temperature"]=segment["temperature"]
        dyn.set_temperature(temperature_K=segment["temperature"])

        print(
            f"{segment['phase']} at {segment['temperature']} K: "
            f"run {remaining_steps} steps "
            f"(global {step_counter['step']} -> {segment_end})"
        )

        dyn.run(remaining_steps)
        completed_steps=segment_end


if __name__=="__main__":
    run()
