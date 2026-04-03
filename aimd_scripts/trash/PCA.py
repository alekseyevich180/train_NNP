import os
import numpy as np
import networkx as nx
from collections import Counter
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase.units import fs, GPa
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
import pfp_api_client
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

# ====================================================================
# [ 1. 全参数化配置区 ]
# ====================================================================
CONFIG = {
    "system": {
        "input_file": "aimd.cif",
        "target_c_count": 9,
        "bottom_z_threshold": 7.50,
        "output_root": "C9_ZnO_1080K"
    },
    "md_control": {
        "initial_temp": 280,
        "final_temp": 1080,
        "timestep": 0.5,        # fs
        "ramp_interval": 100,   # K
        "ramp_steps": 10000,
        "stab_steps": 10000,
        "prod_steps": 500000,
        "tau_t": 100.0          # Berendsen 耦合常数
    },
    "monitoring": {
        "interval": 100,       # 每多少步记录一次
        "archive_size": 10000   # 每个文件夹存放的步数
    },
    "chemistry": {
        # 文献 4 核心：定义指纹识别的成键截断
        "cutoffs": {
            ('C', 'C'): 1.85, ('C', 'O'): 1.65, ('C', 'H'): 1.35, 
            ('O', 'H'): 1.25, ('Zn', 'O'): 2.20, ('Zn', 'C'): 2.30
        }
    }
}

# 全局总步数计数器
total_step_counter = 0

estimator_d3 = Estimator(calc_mode=EstimatorCalcMode.PBE_U_PLUS_D3)
calculator_d3 = ASECalculator(estimator_d3)
print(f"PFP Model Version: {estimator_d3.model_version}")

print(f"pfp_api_client: {pfp_api_client.__version__}")

# ====================================================================
# [ 2. 碎片指纹引擎 (增强吸附分析功能) ]
# ====================================================================
class FragmentEngine:
    @staticmethod
    def get_fingerprints(atoms_obj, cutoffs):
        """将结构转化为碎片指纹，特别关注吸附界面"""
        i, j = neighbor_list('ij', atoms_obj, cutoffs)
        G = nx.Graph()
        G.add_nodes_from(range(len(atoms_obj)))
        G.add_edges_from(zip(i, j))
        
        c_indices = [n for n, a in enumerate(atoms_obj) if a.symbol == 'C']
        seen = set()
        fingerprints = []
        is_adsorbed = False # 标记当前帧是否存在吸附
        
        for idx in c_indices:
            if idx not in seen:
                cluster = nx.node_connected_component(G, idx)
                for node in cluster:
                    symbol = atoms_obj[node].symbol
                    neighbors = [atoms_obj[nb].symbol for nb in G.neighbors(node)]
                    
                    # 识别核心氧化/吸附信号
                    # 1. 羰基氧连了 Zn? (代表吸附)
                    if symbol == 'O' and 'Zn' in neighbors:
                        is_adsorbed = True
                    
                    if symbol in ['C', 'O']:
                        nb_key = "".join([f"{s}{c}" for s, c in sorted(Counter(neighbors).items())])
                        fingerprints.append(f"{symbol}-({nb_key})")
                seen.update(cluster)
        
        return dict(Counter(fingerprints)), is_adsorbed

# ====================================================================
# [ 3. 增强型监测函数 ]
# ====================================================================
# 新增全局变量用于因果分析
adsorption_start_step = None 

def global_monitoring(dyn_obj, atoms_obj, current_temp):
    global total_step_counter, adsorption_start_step
    total_step_counter += 1
    
    cfg_mon = CONFIG["monitoring"]
    
    # 每步都检查吸附状态，但仅定期写入日志
    fp, is_currently_adsorbed = FragmentEngine.get_fingerprints(atoms_obj, CONFIG["chemistry"]["cutoffs"])
    
    # --- 核心逻辑：吸附因果分析 ---
    if is_currently_adsorbed and adsorption_start_step is None:
        adsorption_start_step = total_step_counter
        print(f"DEBUG: Adsorption detected at Step {total_step_counter}")
    elif not is_currently_adsorbed and adsorption_start_step is not None:
        duration = total_step_counter - adsorption_start_step
        adsorption_start_step = None # 重置
    
    # --- 定期记录逻辑 ---
    if total_step_counter % cfg_mon["interval"] == 0:
        save_dir = os.path.join(CONFIG["system"]["output_root"], f"set_{total_step_counter // cfg_mon['archive_size']}")
        os.makedirs(save_dir, exist_ok=True)
        
        epot = atoms_obj.get_potential_energy()
        
        # 专门写入一个用于分析“吸附-反应”关联的简洁日志
        causal_log = os.path.join(CONFIG["system"]["output_root"], "causal_analysis.log")
        with open(causal_log, "a") as f:
            if total_step_counter == cfg_mon["interval"]:
                f.write("Step,Is_Adsorbed,E_pot,C2H2_Count,C1H3_Count\n")
            
            # 提取关键反应指标：长链(C2H2)和断链(C1H3)的数量
            c2h2 = fp.get('C-(C2H2)', 0)
            c1h3 = fp.get('C-(C1H3)', 0)
            f.write(f"{total_step_counter},{int(is_currently_adsorbed)},{epot:.4f},{c2h2},{c1h3}\n")

        # 原有的 monitor.log 保持不变，用于 PCA 数据对齐
        log_path = os.path.join(save_dir, "monitor.log")
        with open(log_path, "a") as f:
            if not os.path.exists(log_path):
                f.write("Step Temp E_pot Fingerprints\n")
            f.write(f"{total_step_counter} {current_temp} {epot:.4f} {str(fp)}\n")
            
# ====================================================================
# [ 4. 模拟主循环 (分阶段控制) ]
# ====================================================================
def run_simulation():
    # 初始化
    atoms = read(CONFIG["system"]["input_file"])
    atoms.calc = calculator_d3  # 请确保外部已定义计算器
    atoms.pbc = True
    
    # 应用约束
    bottom_z = CONFIG["system"]["bottom_z_threshold"]
    fixed_indices = [a.index for a in atoms if a.position[2] < bottom_z]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))
    print(f"Initial: Fixed {len(fixed_indices)} atoms at Z < {bottom_z}")

    # 初始速度
    t_init = CONFIG["md_control"]["initial_temp"]
    MaxwellBoltzmannDistribution(atoms, temperature_K=t_init)
    
    # --- 阶段 0: 初始平衡 ---
    dyn = NVTBerendsen(atoms, timestep=CONFIG["md_control"]["timestep"]*fs, 
                       temperature_K=t_init, taut=CONFIG["md_control"]["tau_t"])
    dyn.attach(lambda d=dyn: global_monitoring(d, atoms, t_init), interval=1)
    print(f">> Balancing at {t_init}K...")
    dyn.run(10000)

    # --- 阶段 1: 阶梯升温 (参数化循环) ---
    curr_t = t_init
    t_final = CONFIG["md_control"]["final_temp"]
    ramp_cfg = CONFIG["md_control"]
    
    while curr_t < t_final:
        curr_t += ramp_cfg["ramp_interval"]
        dyn.set_temperature(temperature_K=curr_t)
        
        # 重新挂载监听器以更新温度标签
        dyn.observers.clear()
        dyn.attach(lambda d=dyn, t=curr_t: global_monitoring(d, atoms, t), interval=1)
        
        print(f">> Ramping to {curr_t}K...")
        dyn.run(ramp_cfg["ramp_steps"])
        dyn.run(ramp_cfg["stab_steps"])

    # --- 阶段 2: 生产跑 (NPT) ---
    print(f">> Production Phase at {t_final}K...")
    dyn_npt = NPT(atoms, timestep=ramp_cfg["timestep"]*fs, temperature_K=t_final,
                  externalstress=0.1e-6*GPa, ttime=100.0, pfactor=None)
    dyn_npt.attach(lambda d=dyn_npt: global_monitoring(d, atoms, t_final), interval=1)
    dyn_npt.run(ramp_cfg["prod_steps"])

if __name__ == "__main__":
    run_simulation()