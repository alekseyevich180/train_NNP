import os
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================================================
# [ 1. 配置区 ]
# ====================================================================
PLOT_CONFIG = {
    "data": {
        "root_dir": "experiment_C9_ZnO_1080K",
        "log_filename": "monitor.log"
    },
    "selection": {
        "targets": ['C-(C2H2)', 'C-(C1H3)', 'C-(H4)', 'O-(C1Zn1)', 'O-(H1Zn1)']
    },
    "vis": {
        "figsize": (12, 6),
        "colors": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        "style": "seaborn-v0_8-whitegrid"
    }
}

# ====================================================================
# [ 2. 数据处理 ]
# ====================================================================
def get_processed_df(root_dir):
    search_pattern = os.path.join(root_dir, "set_*", PLOT_CONFIG["data"]["log_filename"])
    log_files = sorted(glob.glob(search_pattern))
    
    all_data = []
    for path in log_files:
        with open(path, 'r') as f:
            lines = f.readlines()[1:] 
            for line in lines:
                parts = line.strip().split(maxsplit=4)
                if len(parts) < 5: continue
                try:
                    row = {"Step": int(parts[0]), "Energy": float(parts[2]), **ast.literal_eval(parts[4])}
                    all_data.append(row)
                except: continue
    return pd.DataFrame(all_data).fillna(0).sort_values("Step")

# ====================================================================
# [ 3. 绘图逻辑 ]
# ====================================================================
def run_dual_plot_analysis(df):
    plt.style.use(PLOT_CONFIG["vis"]["style"])
    available_targets = [t for t in PLOT_CONFIG["selection"]["targets"] if t in df.columns]

    # --- 图 1: 碎片数量 vs. 步数 (时间演化) ---
    plt.figure(figsize=PLOT_CONFIG["vis"]["figsize"])
    for i, frag in enumerate(available_targets):
        plt.plot(df["Step"], df[frag], label=frag, color=PLOT_CONFIG["vis"]["colors"][i], linewidth=2)
    
    plt.xlabel("MD Total Steps", fontsize=12)
    plt.ylabel("Fragment Count", fontsize=12)
    plt.title("Figure A: Fragment Population Evolution Over Time", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- 图 2: 碎片数量 vs. 能量 (状态空间) ---
    plt.figure(figsize=PLOT_CONFIG["vis"]["figsize"])
    # 按能量对数据进行排序，以便绘制平滑趋势或散点趋势
    df_sorted_energy = df.sort_values("Energy")
    
    for i, frag in enumerate(available_targets):
        # 使用散点图配合轻微透明度，可以观察不同能量区间内碎片的分布密度
        plt.scatter(df_sorted_energy["Energy"], df_sorted_energy[frag], 
                    label=frag, color=PLOT_CONFIG["vis"]["colors"][i], s=25, alpha=0.5)
    
    plt.xlabel("Potential Energy (eV)", fontsize=12)
    plt.ylabel("Fragment Count", fontsize=12)
    plt.title("Figure B: Fragment Distribution vs. System Potential Energy", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ROOT = PLOT_CONFIG["data"]["root_dir"]
    if os.path.exists(ROOT):
        data_df = get_processed_df(ROOT)
        run_dual_plot_analysis(data_df)