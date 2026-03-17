#!/bin/sh
#SBATCH -J uma_test
#SBATCH -p L1cpu
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH -t 120:00:00
#SBATCH --exclusive

ulimit -s unlimited
module purge
module load oneapi_compiler/2023.0.0 oneapi_mkl/2023.0.0 openmpi/4.1.5-oneapi-2023.0.0-classic

export KMP_STACKSIZE=512m
export UCX_TLS='self,sm,ud'

# ✅ 激活 conda 环境
source ~/miniforge3/bin/activate uma_env

# 👉 如果你有 API key（用 UMA 才需要）
# export PFP_API_KEY=xxxx

set -euo pipefail

cd /home/你的路径/UMA-campare/interface/LiPSCl-400K

###########################################
ENSEMBLE=npt
TEMPERATURE=400
###########################################

python -u md.py \
    --atoms_path LiLPSC110_2x2x1.cif \
    --out_traj_path UMA-LiPSCl_${ENSEMBLE^^}_$TEMPERATURE.traj \
    --temperature $TEMPERATURE \
    --timestep 1 \
    --run_steps 100000 \
    --traj_interval 100 \
    --ensemble $ENSEMBLE \
    --uma_model uma-s-1p1 \
    --device cpu \
    --taut 100 \
    --pressure 1.0 \
    --taup 1000 \
    --package ocp | tee UMA-LiPSCl_${ENSEMBLE^^}_$TEMPERATURE.log