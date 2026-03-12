import os
import numpy as np
import networkx as nx
from collections import Counter
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase.units import fs, GPa
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from sklearn.decomposition import PCA
import pfp_api_client
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

# ====================================================================
# [ 1. 全参数化配置区 ] - 已更新梯度升温与大体系优化参数
# ====================================================================
CONFIG = {
    "system": {
        "input_file": "aimd_bigger.cif",
        "bottom_z_threshold": 9.00,
        "output_root": "C9_ZnO_1080K_AL_bigger"
    },
    "active_learning": {
        "enabled": True,
        # 阈值建议：针对大体系力偏差，0.1-0.15排除稳态，0.3-0.5捕获过渡态 [cite: 82, 176]
        "sigma_low": 0.12,      
        "sigma_high": 0.45,     
        "pca_dimensions": 12,   
        "val_dir": "deepmd_validation_set", 
        "cif_dir": "cif_frames_per_100",    
        "save_interval_limit": 100          
    },
    "md_control": {
        # --- 梯度升温控制参数 (Ramping Control) ---
        "initial_temp": 280,    
        "final_temp": 1080,     
        "ramp_interval": 100,   
        "ramp_steps": 20000,    
        "stab_steps": 10000,    
        
        # --- 生产跑参数 ---
        "prod_steps": 3000000,   
        "timestep": 0.5,        
        "tau_t": 100.0          
    },
    "chemistry": {
        "cutoffs": {
            # 这里的截断值定义了指纹识别如何“看”氧化反应 [cite: 493, 552]
            ('C', 'C'): 1.85, ('C', 'O'): 1.65, ('C', 'H'): 1.35, 
            ('O', 'H'): 1.25, ('Zn', 'O'): 2.20, ('Zn', 'C'): 2.30
        }
    }
}
# ====================================================================
# [ 2. 主动学习筛选器 - 修改版 ]
# ====================================================================
# ====================================================================
# [ 2. 主动学习筛选器 - 增强分文件夹版 ]
# ====================================================================
class ActiveLearningSieve:
    def __init__(self, config):
        self.cfg = config
        self.output_root = CONFIG["system"]["output_root"]
        self.val_dir_base = config["val_dir"]  # 验证集根目录
        self.cif_dir_base = config["cif_dir"]  # CIF 根目录
        
        # 专门为验证集准备的内部计数器
        self.al_save_count = 0 
        self.last_al_save_step = -100

    def run(self, atoms, step, fp):
        # 1. 模拟不确定性评估
        try:
            # 这里的 sigma 逻辑在对接 DeepMD 时应替换为真实模型偏差
            sigma = atoms.get_potential_energy() % 0.4 
        except:
            sigma = 0.0

        # 2. 定时生成 CIF 文件 (每 100 步触发，按步数区间分文件夹)
        if step % 100 == 0:
            cif_set_idx = step // 10000
            current_cif_set_dir = os.path.join(self.output_root, self.cif_dir_base, f"set_{cif_set_idx}")
            os.makedirs(current_cif_set_dir, exist_ok=True)
            
            cif_path = os.path.join(current_cif_set_dir, f"step_{step}.cif")
            write(cif_path, atoms)

        # 3. 主动学习筛选逻辑 (按保存数量分文件夹)
        if self.cfg["sigma_low"] < sigma < self.cfg["sigma_high"]:
            if step - self.last_al_save_step >= self.cfg["save_interval_limit"]:
                # --- 新增：计算验证集的 set 编号 ---
                # 每 100 个结构存入一个文件夹
                val_set_idx = self.al_save_count // 100
                current_val_set_dir = os.path.join(self.output_root, self.val_dir_base, f"set_{val_set_idx}")
                os.makedirs(current_val_set_dir, exist_ok=True)
                
                # 文件名保持包含步数和 sigma，方便溯源
                xyz_path = os.path.join(current_val_set_dir, f"AL_step_{step}_sig_{sigma:.3f}.xyz")
                
                atoms.info['fingerprints'] = str(fp)
                write(xyz_path, atoms)
                
                # 更新计数器和状态
                self.al_save_count += 1
                self.last_al_save_step = step
                print(f"[*] AL 筛选: 保存第 {self.al_save_count} 个关键结构 -> {xyz_path}")
# ====================================================================
# [ 3. 碎片指纹识别引擎 ]
# ====================================================================
class FragmentEngine:
    @staticmethod
    def get_fingerprints(atoms_obj, cutoffs):
        """基于图论的碎片识别 [cite: 96, 552]"""
        i, j = neighbor_list('ij', atoms_obj, cutoffs)
        G = nx.Graph()
        G.add_nodes_from(range(len(atoms_obj)))
        G.add_edges_from(zip(i, j))
        
        c_indices = [n for n, a in enumerate(atoms_obj) if a.symbol == 'C']
        seen = set()
        fingerprints = []
        is_adsorbed = False 
        
        for idx in c_indices:
            if idx not in seen:
                cluster = nx.node_connected_component(G, idx)
                for node in cluster:
                    symbol = atoms_obj[node].symbol
                    neighbors = [atoms_obj[nb].symbol for nb in G.neighbors(node)]
                    # 识别吸附信号：氧连 Zn [cite: 493]
                    if symbol == 'O' and 'Zn' in neighbors:
                        is_adsorbed = True
                    if symbol in ['C', 'O']:
                        nb_key = "".join([f"{s}{c}" for s, c in sorted(Counter(neighbors).items())])
                        fingerprints.append(f"{symbol}-({nb_key})")
                seen.update(cluster)
        return dict(Counter(fingerprints)), is_adsorbed

# ====================================================================
# [ 4. 全局监测回调 ]
# ====================================================================
total_step_counter = 0
sieve = ActiveLearningSieve(CONFIG["active_learning"])

def global_monitoring(dyn_obj, atoms_obj, temp):
    global total_step_counter
    total_step_counter += 1
    
    # 提取指纹
    fp, is_ads = FragmentEngine.get_fingerprints(atoms_obj, CONFIG["chemistry"]["cutoffs"])
    
    # 执行筛选与结构生成
    sieve.run(atoms_obj, total_step_counter, fp)
    
    # 记录文本日志 (每 100 步)
    if total_step_counter % 100 == 0:
        log_path = os.path.join(CONFIG["system"]["output_root"], "al_monitor.log")
        is_new = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if is_new:
                f.write("Step Temp Adsorbed Fingerprints\n")
            f.write(f"{total_step_counter:<8d} {temp:<6d} {int(is_ads):<10d} {str(fp)}\n")

# ====================================================================
# [ 5. 修复后的模拟主程序：引入梯度升温控制 ]
# ====================================================================
def run():
    # 1. 初始化计算器与结构
    estimator = Estimator(calc_mode=EstimatorCalcMode.PBE_U_PLUS_D3)
    calculator = ASECalculator(estimator)
    
    atoms = read(CONFIG["system"]["input_file"])
    atoms.calc = calculator
    atoms.pbc = True
    
    # 2. 设置底部约束 (针对 660 个原子的 ZnO 表面体系)
    fixed = [a.index for a in atoms if a.position[2] < CONFIG["system"]["bottom_z_threshold"]]
    atoms.set_constraint(FixAtoms(indices=fixed))

    # 3. 获取升温控制参数
    ctrl = CONFIG["md_control"]
    curr_t = ctrl.get("initial_temp", 280) # 建议从 280K 开始
    t_final = ctrl.get("final_temp", 1080)
    
    # 初始速度分配
    MaxwellBoltzmannDistribution(atoms, temperature_K=curr_t)
    
    # 4. 建立 NVT 动力学对象
    dyn = NVTBerendsen(atoms, timestep=ctrl["timestep"]*fs, 
                       temperature_K=curr_t, taut=ctrl["tau_t"])
    
    # --- 阶段 A: 初始低温平衡 (防止大体系初始化崩溃) ---
    print(f">> 正在进行初始平衡 (温度: {curr_t}K)...")
    # 绑定监控器，传递当前温度标签
    dyn.attach(lambda d=dyn, t=curr_t: global_monitoring(d, atoms, t), interval=1)
    dyn.run(10000)

    # --- 阶段 B: 梯度升温循环 (符合论文提到的 multistep 过程) ---
    # 参考论文中的升温思路，逐步接近反应条件 [cite: 78, 80]
    while curr_t < t_final:
        curr_t += ctrl["ramp_interval"]
        if curr_t > t_final: curr_t = t_final
        
        # 更新控温器目标温度
        dyn.set_temperature(temperature_K=curr_t)
        
        # 重要：清除旧监控器并挂载带有新温度标签的监控器
        dyn.observers.clear()
        dyn.attach(lambda d=dyn, t=curr_t: global_monitoring(d, atoms, t), interval=1)
        
        print(f">> 梯度升温中: {curr_t}K (升温 {ctrl['ramp_steps']} 步 + 稳定 {ctrl['stab_steps']} 步)...")
        dyn.run(ctrl["ramp_steps"]) # 升温阶段
        dyn.run(ctrl["stab_steps"]) # 稳定阶段

    # --- 阶段 C: 长期生产跑 (用于主动学习筛选氧化反应结构) ---
    print(f">> 达到最终温度 {t_final}K，开始执行长期生产跑 ({ctrl['prod_steps']} 步)...")
    dyn.run(ctrl["prod_steps"])

if __name__ == "__main__":
    run()