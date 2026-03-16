#!/bin/bash
#PJM -L rscgrp=a-pj24001724
#PJM -L node=1
#PJM --mpi proc=1
#PJM -L elapse=120:00:00
#PJM -j

set -euo pipefail

# ==========================
# 基本设置
# ==========================

SCRIPT_NAME="wu_deep.sh"

# 120小时作业时长，预留4小时buffer给收尾和重投
JOB_HOURS=120
BUFFER_HOURS=4
MAX_TIME=$(((JOB_HOURS - BUFFER_HOURS) * 3600))

# ==========================
# 环境加载
# ==========================

module load intel impi lammps
source /home/pj24001724/ku40000345/wu/deepmd_kit/use.sh

# ==========================
# 路径设置
# ==========================

TRAIN_DIR="/home/pj24001724/ku40000345/wu/deepmd_train/surface_600_1200k/01.train"
INPUT_JSON="${TRAIN_DIR}/input.json"

LMP_DIR="/home/pj24001724/ku40000345/wu/deepmd_train/surface_600_1200k/02.lmp"
LMP_EXE="/home/pj24001724/ku40000345/wu/deepmd_kit/deepmd_root/bin/lmp"

# ==========================
# 线程设置
# ==========================

export OMP_NUM_THREADS=15
export DP_INTRA_OP_PARALLELISM_THREADS=15
export DP_INTER_OP_PARALLELISM_THREADS=1

export TF_INTRA_OP_PARALLELISM_THREADS=15
export TF_INTER_OP_PARALLELISM_THREADS=1

export TF_CPP_MIN_LOG_LEVEL=3

# ==========================
# 数据抽取
# ==========================

DIR1="/home/pj24001724/ku40000345/wu/deepmd_train/surface_600k/puresurface_300k"
DIR2="/home/pj24001724/ku40000345/wu/deepmd_train/surface_600k/puresurface_800k/puresurface_800k/dp_data_aimd1"
DIR3="/home/pj24001724/ku40000345/wu/deepmd_train/surface_600k/puresurface_1200k/puresurface_1200k/dp_data_aimd1"

DEST="/home/pj24001724/ku40000345/wu/deepmd_train/surface_600_1200k/01.train_val"

NUM=50
SOURCES=("$DIR1" "$DIR2" "$DIR3")

if [ ! -d "$DEST/set.000" ]; then

    echo ">> 抽取验证集数据..."

    mkdir -p "$DEST"

    GLOBAL_IDX=0

    for src in "${SOURCES[@]}"; do

        if [ ! -d "$src" ]; then
            echo "warning: $src 不存在"
            continue
        fi

        selected_dirs=$(find "$src" -maxdepth 1 -mindepth 1 -type d -name "set.*" | shuf -n "$NUM")

        for sub_dir in $selected_dirs; do

            formatted_idx=$(printf "%03d" "$GLOBAL_IDX")

            cp -r "$sub_dir" "$DEST/set.$formatted_idx"

            GLOBAL_IDX=$((GLOBAL_IDX + 1))

        done

    done

else
    echo ">> 验证集已存在"
fi

# 清理 type.raw
rm -f ${TRAIN_DIR}/set.*/type.raw
rm -f ${TRAIN_DIR}/set.*/type_map.raw

rm -f ${DEST}/set.*/type.raw
rm -f ${DEST}/set.*/type_map.raw

# ==========================
# 开始训练
# ==========================

OUTDIR="${TRAIN_DIR}/run.train"
mkdir -p "${OUTDIR}"

cd "${OUTDIR}"

echo "==================== START TRAINING ===================="

set +e

if [ -f "model.ckpt.index" ]; then

    echo ">> restart training"

    timeout ${MAX_TIME}s dp train --restart model.ckpt input.json

else

    echo ">> start new training"

    cp -f "${INPUT_JSON}" ./input.json

    timeout ${MAX_TIME}s dp train input.json

fi

EXIT_CODE=$?

set -e

# ==========================
# 判断状态
# ==========================

RESUBMIT=0

if [ $EXIT_CODE -eq 124 ]; then

    echo ">> timeout reached"

    RESUBMIT=1

elif [ $EXIT_CODE -eq 0 ]; then

    echo ">> training finished"

else

    echo ">> training crashed"
    exit 1

fi

# ==========================
# Freeze & Test
# ==========================

if [ $RESUBMIT -eq 0 ]; then

    echo "==================== FREEZE MODEL ===================="

    dp freeze -o graph.pb
    dp compress -i graph.pb -o compress.pb

    echo "==================== DP TEST ===================="

    dp test -m graph.pb -s "${TRAIN_DIR}" -n 100 > test_results.log

    if [ -d "${LMP_DIR}" ] && [ -f "${LMP_DIR}/in.lammps" ]; then

        echo ">> running LAMMPS"

        cp compress.pb "${LMP_DIR}/compress.pb"
        cp graph.pb "${LMP_DIR}/graph.pb"

        cd "${LMP_DIR}"

        "${LMP_EXE}" -in in.lammps

    else

        echo "LAMMPS input missing"

    fi

else

    echo ">> training not finished, skip freeze"

fi

echo "==================== CURRENT JOB DONE ===================="

# ==========================
# 自动重新提交
# ==========================

if [ $RESUBMIT -eq 1 ]; then

    echo ">> resubmitting job..."

    cd "${PJM_O_WORKDIR:-$PWD}"

    if [ -f "${SCRIPT_NAME}" ]; then

        qsub "${SCRIPT_NAME}"

        echo ">> next job submitted"

    else

        echo ">> ERROR: cannot find ${SCRIPT_NAME}"

    fi

fi
