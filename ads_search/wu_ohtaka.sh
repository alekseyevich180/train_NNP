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
SURFACE=surface-3O.cif
MOLECULES="5-ketone.cif 2-6ketone.cif 3-6ketone.cif"
N_TRIALS=300
FMAX=1e-4
UMA_MODEL=uma-s-1p1
DEVICE=cpu
INCLUDE_D3=0
OUTPUT_DIR=output
LOG_NAME=ads_search.log
###########################################

CMD="python -u ads.py \
    --surface $SURFACE \
    --molecules $MOLECULES \
    --n_trials $N_TRIALS \
    --fmax $FMAX \
    --uma_model $UMA_MODEL \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR"

if [ "$INCLUDE_D3" = "1" ]; then
    CMD=\"$CMD --include_d3\"
fi

eval "$CMD" | tee "$LOG_NAME"
