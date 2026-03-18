#!/bin/bash
#SBATCH -J test
#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0:30:00
#SBATCH --exclusive

set -euo pipefail

ulimit -s unlimited
module purge
module load oneapi_compiler/2023.0.0 oneapi_mkl/2023.0.0 openmpi/4.1.5-oneapi-2023.0.0-classic

export KMP_STACKSIZE=512m
export UCX_TLS='self,sm,ud'

source /home/k0710/k071001/miniforge3/etc/profile.d/conda.sh
conda activate /home/k0710/k071001/UMA/uma_clean

# 👉 如果你有 API key（用 UMA 才需要）
# export PFP_API_KEY=xxxx

cd /home/k0710/k071001/UMA/lin_test

###########################################
SURFACE=surface.vasp
MOLECULES="G.vasp H.vasp S.vasp"
N_TRIALS=300
FMAX=1e-4
UMA_MODEL=uma-s-1p2
CHECKPOINT=/home/k0710/k071001/UMA/checkpoints/uma-s-1p2.pt
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
    --checkpoint $CHECKPOINT \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR"

if [ "$INCLUDE_D3" = "1" ]; then
    CMD="$CMD --include_d3"
fi

eval "$CMD" | tee "$LOG_NAME"
