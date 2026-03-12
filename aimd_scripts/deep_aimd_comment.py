import os
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
import pfp_api_client
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

# ------------------------------
# 全局参数配置
# ------------------------------
CONFIG={
"system":{
"input_file":"aimd_bigger.cif", # 初始结构
"bottom_z_threshold":9.0,       # z 小于该值的原子固定（表面底层固定）
"output_root":"AIMD_dataset"    # 输出总目录
},
"md_control":{
"initial_temp":280,             # 初始温度
"final_temp":1080,              # 最终温度
"ramp_interval":100,            # 每次升温步长
"ramp_steps":20000,             # 每次升温后的 MD 步数
"stab_steps":10000,             # 稳定步骤
"prod_steps":3000000,           # 生产阶段 MD 步数
"timestep":0.5,                 # MD timestep (fs)
"tau_t":100.0                   # thermostat 参数
},
"output":{
"save_interval":100,            # 每多少 MD step 保存一次结构
"deepmd_set_size":1000,         # 每个 set.xxx 包含多少 frames
"deepmd_dir":"deepmd_dataset",  # DeepMD 数据目录
"cif_dir":"cif_frames"          # CIF 输出目录（用于 PCA / 可视化）
}
}

# ------------------------------
# DeepMD dataset writer
# 将 MD 轨迹转换为 DeepMD 格式
# ------------------------------
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

    # 写入 type.raw 与 type_map.raw
    # DeepMD 用于识别元素类型
    def write_type_files(self,atoms):
        symbols=atoms.get_chemical_symbols()
        uniq=sorted(set(symbols))
        type_map={s:i for i,s in enumerate(uniq)}
        type_list=[type_map[s] for s in symbols]
        np.savetxt(os.path.join(self.root,"type.raw"),np.array(type_list),fmt="%d")
        with open(os.path.join(self.root,"type_map.raw"),"w") as f:
            for s in uniq:
                f.write(s+"\n")

    # 添加一帧 MD 数据
    def add_frame(self,atoms):
        self.coord.append(atoms.get_positions().reshape(-1))   # 原子坐标
        self.force.append(atoms.get_forces().reshape(-1))       # 原子力
        self.energy.append(atoms.get_potential_energy())        # 总能量
        self.box.append(atoms.get_cell().array.reshape(-1))     # 晶胞
        self.frame+=1
        if self.frame>=self.set_size:
            self.write_set()
            self.reset()

    # 清空缓存
    def reset(self):
        self.coord=[]
        self.force=[]
        self.energy=[]
        self.box=[]
        self.frame=0

    # 写入 set.xxx
    def write_set(self):
        set_dir=os.path.join(self.root,f"set.{self.set_id:03d}")
        os.makedirs(set_dir,exist_ok=True)
        np.save(os.path.join(set_dir,"coord.npy"),np.array(self.coord))
        np.save(os.path.join(set_dir,"force.npy"),np.array(self.force))
        np.save(os.path.join(set_dir,"energy.npy"),np.array(self.energy))
        np.save(os.path.join(set_dir,"box.npy"),np.array(self.box))
        print(f"write {set_dir}")
        self.set_id+=1

# ------------------------------
# 主程序
# ------------------------------
def run():
    root=CONFIG["system"]["output_root"]
    deepmd_root=os.path.join(root,CONFIG["output"]["deepmd_dir"])
    cif_root=os.path.join(root,CONFIG["output"]["cif_dir"])
    os.makedirs(cif_root,exist_ok=True)

    # 读取结构
    atoms=read(CONFIG["system"]["input_file"])

    # Matlantis calculator（保持不修改）
    estimator=Estimator(calc_mode=EstimatorCalcMode.PBE_U_PLUS_D3)
    calculator=ASECalculator(estimator)
    atoms.calc=calculator
    atoms.pbc=True

    # 固定底层原子
    fixed=[a.index for a in atoms if a.position[2]<CONFIG["system"]["bottom_z_threshold"]]
    atoms.set_constraint(FixAtoms(indices=fixed))

    # 初始化 DeepMD writer
    writer=DeepMDWriter(atoms,deepmd_root,CONFIG["output"]["deepmd_set_size"])

    ctrl=CONFIG["md_control"]
    curr_t=ctrl["initial_temp"]
    final_t=ctrl["final_temp"]

    # 初始化速度
    MaxwellBoltzmannDistribution(atoms,temperature_K=curr_t)

    # NVT Berendsen thermostat
    dyn=NVTBerendsen(atoms,timestep=ctrl["timestep"]*fs,temperature_K=curr_t,taut=ctrl["tau_t"])

    step_counter={"step":0}

    # 保存 frame 的函数
    def save_frame():
        step_counter["step"]+=1
        if step_counter["step"]%CONFIG["output"]["save_interval"]!=0:
            return
        writer.add_frame(atoms) # DeepMD 数据
        cif_file=os.path.join(cif_root,f"frame_{step_counter['step']:08d}.cif")
        write(cif_file,atoms)   # CIF 用于 PCA / 可视化

    dyn.attach(save_frame,interval=1)

    print(f"Initial equilibration at {curr_t} K")
    dyn.run(10000)

    # ------------------------------
    # 梯度升温阶段
    # ------------------------------
    while curr_t<final_t:
        curr_t+=ctrl["ramp_interval"]
        if curr_t>final_t:
            curr_t=final_t
        dyn.set_temperature(temperature_K=curr_t)
        print(f"Ramp to {curr_t} K")
        dyn.run(ctrl["ramp_steps"])
        dyn.run(ctrl["stab_steps"])

    # ------------------------------
    # 生产阶段 MD
    # ------------------------------
    print(f"Production run at {final_t} K")
    dyn.run(ctrl["prod_steps"])

if __name__=="__main__":
    run()