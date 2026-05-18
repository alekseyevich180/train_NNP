#!/bin/bash
#PJM -L rscgrp=a-pj24001724
#PJM -L node=1
#PJM --mpi proc=1
#PJM -L elapse=168:00:00
#PJM -j

set -euo pipefail

# ==================== 关键设置 ====================
SCRIPT_NAME="wu_deep.sh"       # 必须与本脚本文件名一致
MAX_TIME=594000                # 165h，给自动重提留缓冲

# qsub 在本机环境下不会把 "qsub wu_deep.sh v3 ..." 的位置参数传进脚本。
# 因此无参数提交时默认跑当前推荐任务；仍可用 pjsub 参数或环境变量覆盖。
DEFAULT_TRAIN_ID="v4"
DEFAULT_INPUT_NAME="input.json"
DEFAULT_BASE_CKPT="run.train/model.ckpt"
DEFAULT_TRAIN_MODE="restart"
# =================================================

# 用法：
#   qsub wu_deep.sh
#       使用默认设置：input.json + run.train/model.ckpt restart，输出到 run.train.v4
#
#   qsub wu_deep.sh v3 input.v2 run.train/model.ckpt init
#       注意：部分 qsub 不传位置参数；若输出仍显示 TRAIN_ID=v3，说明参数生效。
#
#   qsub -v TRAIN_ID=v4,INPUT_NAME=input.v2,BASE_CKPT=run.train.v3/model.ckpt,TRAIN_MODE=init wu_deep.sh
#       qsub 更稳的改参方式：用环境变量覆盖默认设置。
#
#   pjsub wu_deep.sh v2 input.v2 run.train/model.ckpt
#       使用 input.v2，从 run.train/model.ckpt 初始化新训练，输出到 run.train.v2
#
#   pjsub wu_deep.sh v3 input.v2 run.train/model.ckpt init
#       推荐用于重新适应新数据：用旧模型初始化权重，但重置 step/lr，输出到 run.train.v3
#
#   pjsub wu_deep.sh v3 input.v2 run.train.v2/model.ckpt restart
#       仅用于真的想延续上一轮 step/lr 时使用

# 1) 环境加载
module load intel impi lammps
source /home/pj24001724/ku40000345/wu/deepmd_kit/use.sh

# 2) 路径设置（按你当前 nnp_train）
ROOT_DIR="/home/pj24001724/ku40000345/wu/deepmd_train/nnp_train"

TRAIN_ID="${TRAIN_ID:-${1:-${DEFAULT_TRAIN_ID}}}"
INPUT_NAME="${INPUT_NAME:-${2:-${DEFAULT_INPUT_NAME}}}"
BASE_CKPT="${BASE_CKPT:-${3:-${DEFAULT_BASE_CKPT}}}"
TRAIN_MODE="${TRAIN_MODE:-${4:-${DEFAULT_TRAIN_MODE}}}"
DP_TEST_N="${DP_TEST_N:-all}"

case "${INPUT_NAME}" in
    /*) INPUT_JSON="${INPUT_NAME}" ;;
    *) INPUT_JSON="${ROOT_DIR}/${INPUT_NAME}" ;;
esac

if [ "${TRAIN_ID}" = "train" ]; then
    OUTDIR="${ROOT_DIR}/run.train"
else
    OUTDIR="${ROOT_DIR}/run.train.${TRAIN_ID}"
fi

if [ -n "${BASE_CKPT}" ]; then
    case "${BASE_CKPT}" in
        /*) BASE_CKPT_PATH="${BASE_CKPT}" ;;
        *) BASE_CKPT_PATH="${ROOT_DIR}/${BASE_CKPT}" ;;
    esac
else
    BASE_CKPT_PATH=""
fi

case "${TRAIN_MODE}" in
    init|restart) ;;
    *)
        echo ">> [ERROR] TRAIN_MODE 只能是 init 或 restart，当前为: ${TRAIN_MODE}"
        exit 1
        ;;
esac

# 可选：LAMMPS 测试目录（没有就自动跳过）
LMP_DIR="/home/pj24001724/ku40000345/wu/deepmd_train/nnp_train/02.lmp"
LMP_EXE="/home/pj24001724/ku40000345/wu/deepmd_kit/deepmd_root/bin/lmp"

# 3) 并行设置
NTHREADS=15
export OMP_NUM_THREADS=${NTHREADS}
export DP_INTRA_OP_PARALLELISM_THREADS=${NTHREADS}
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=${NTHREADS}
export TF_INTER_OP_PARALLELISM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3

echo ">> TRAIN_ID=${TRAIN_ID}"
echo ">> INPUT_JSON=${INPUT_JSON}"
echo ">> OUTDIR=${OUTDIR}"
echo ">> TRAIN_MODE=${TRAIN_MODE}"
echo ">> DP_TEST_N=${DP_TEST_N} (all 表示按每个体系 frame 数全量测试)"
if [ -n "${BASE_CKPT_PATH}" ]; then
    echo ">> BASE_CKPT=${BASE_CKPT_PATH}"
else
    echo ">> BASE_CKPT=(none)"
fi

# 4) 输入检查
[ -f "${INPUT_JSON}" ] || { echo "[ERROR] input 不存在: ${INPUT_JSON}"; exit 1; }

if [ -n "${BASE_CKPT_PATH}" ] && [ ! -f "${BASE_CKPT_PATH}.index" ]; then
    echo ">> [ERROR] 指定的基础 checkpoint 不存在: ${BASE_CKPT_PATH}.index"
    exit 1
fi

python3 - "${INPUT_JSON}" << 'PY'
import json, sys
from pathlib import Path

inp = Path(sys.argv[1])
data = json.loads(inp.read_text())

def check_systems(key):
    arr = data["training"][key]["systems"]
    if not arr:
        raise RuntimeError(f"{key}.systems 为空")
    for s in arr:
        p = Path(s)
        if not p.is_dir():
            raise RuntimeError(f"系统目录不存在: {p}")
        has_set = any(x.is_dir() and x.name.startswith("set") for x in p.iterdir())
        if not has_set:
            raise RuntimeError(f"系统目录缺少 set*: {p}")
        if not (p / "type.raw").exists():
            raise RuntimeError(f"系统目录缺少 type.raw: {p}")

check_systems("training_data")
check_systems("validation_data")
print("[OK] input systems 检查通过")
PY

# 5) 进入训练目录
mkdir -p "${OUTDIR}"
cd "${OUTDIR}"
cp -f "${INPUT_JSON}" ./input.json

echo ">> [STEP 1] Start DeepMD training"

set +e
if [ -f "model.ckpt.index" ]; then
    echo ">> 检测到当前目录 model.ckpt，继续本轮断点续训..."
    timeout ${MAX_TIME}s dp train --restart model.ckpt ./input.json
elif [ -n "${BASE_CKPT_PATH}" ] && [ "${TRAIN_MODE}" = "init" ]; then
    echo ">> 本轮首次训练，用基础模型初始化权重并重置 step/lr: ${BASE_CKPT_PATH}"
    timeout ${MAX_TIME}s dp train --init-model "${BASE_CKPT_PATH}" ./input.json
elif [ -n "${BASE_CKPT_PATH}" ]; then
    echo ">> 本轮首次训练，从基础模型断点继续: ${BASE_CKPT_PATH}"
    timeout ${MAX_TIME}s dp train --restart "${BASE_CKPT_PATH}" ./input.json
else
    echo ">> 本轮首次训练，不使用上一轮 checkpoint，从头训练..."
    timeout ${MAX_TIME}s dp train ./input.json
fi
EXIT_CODE=$?
set -e

RESUBMIT=0
if [ ${EXIT_CODE} -eq 124 ]; then
    echo ">> [INFO] 达到单次作业时长上限，准备自动重提"
    RESUBMIT=1
elif [ ${EXIT_CODE} -eq 0 ]; then
    echo ">> [INFO] 训练已完成"
else
    echo ">> [ERROR] 训练异常退出，code=${EXIT_CODE}"
    exit 1
fi

# 6) 仅在训练完整结束后执行
if [ ${RESUBMIT} -eq 0 ]; then
    echo ">> [STEP 2] Freeze & Compress"
    dp freeze -o graph.pb
    dp compress -i graph.pb -o compress.pb

    echo ">> [STEP 3] dp test (按 training_data 和 validation_data 逐体系判断)"
    python3 - "${INPUT_JSON}" > systems.list << 'PY'
import json, sys
from pathlib import Path
import numpy as np

data = json.loads(Path(sys.argv[1]).read_text())
for group in ("training_data", "validation_data"):
    for s in data["training"][group]["systems"]:
        p = Path(s)
        nframes = 0
        for sub in p.iterdir():
            coord = sub / "coord.npy"
            if sub.is_dir() and sub.name.startswith("set") and coord.exists():
                nframes += int(np.load(coord, mmap_mode="r").shape[0])
        print(group, s, nframes)
PY

    : > test_results.log
    while read -r GROUP SYS NFRAMES; do
        echo "==== dp test (${GROUP}, nframes=${NFRAMES}): ${SYS} ====" | tee -a test_results.log
        if [ "${DP_TEST_N}" = "all" ]; then
            dp test -m graph.pb -s "${SYS}" -n "${NFRAMES}" >> test_results.log 2>&1
        else
            dp test -m graph.pb -s "${SYS}" -n "${DP_TEST_N}" >> test_results.log 2>&1
        fi
    done < systems.list

    python3 - test_results.log test_summary.tsv << 'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

rows = []
current = None
energy_per_atom = None
force_rmse = None

section_re = re.compile(r"^==== dp test \(([^,]+), nframes=([0-9]+)\): (.+) ====$")
energy_re = re.compile(r"Energy RMSE/Natoms\s*:\s*([0-9.eE+-]+)")
force_re = re.compile(r"Force\s+RMSE\s*:\s*([0-9.eE+-]+)")

def flush():
    if current and energy_per_atom is not None and force_rmse is not None:
        group, nframes, system = current
        rows.append((group, system, nframes, energy_per_atom, force_rmse))

for line in log_path.read_text(errors="replace").splitlines():
    m = section_re.match(line)
    if m:
        flush()
        current = (m.group(1), int(m.group(2)), m.group(3))
        energy_per_atom = None
        force_rmse = None
        continue
    m = energy_re.search(line)
    if m:
        energy_per_atom = float(m.group(1))
        continue
    m = force_re.search(line)
    if m and force_rmse is None:
        force_rmse = float(m.group(1))

flush()

with out_path.open("w") as f:
    f.write("group\tsystem\tnframes\tenergy_rmse_per_atom_eV\tforce_rmse_eV_per_A\n")
    for row in rows:
        f.write("%s\t%s\t%d\t%.8e\t%.8e\n" % row)

print(f">> [INFO] dp test summary: {out_path}")
PY

    # 可选 LAMMPS
    if [ -d "${LMP_DIR}" ] && [ -f "${LMP_DIR}/in.lammps" ]; then
        echo ">> [STEP 4] 同步势函数并运行 LAMMPS"
        cp -f "${OUTDIR}/compress.pb" "${LMP_DIR}/compress.pb"
        cp -f "${OUTDIR}/graph.pb" "${LMP_DIR}/graph.pb"
        cd "${LMP_DIR}"
        "${LMP_EXE}" -in in.lammps
    else
        echo ">> [INFO] 未找到 LAMMPS 输入，跳过"
    fi
else
    echo ">> [INFO] 本次未完成全部步数，跳过 freeze/test/lammps"
fi

echo "==================== CURRENT JOB DONE ===================="

# 7) 自动重提
if [ ${RESUBMIT} -eq 1 ]; then
    cd "${PJM_O_WORKDIR:-${PBS_O_WORKDIR:-${ROOT_DIR}}}"
    if [ -f "${SCRIPT_NAME}" ]; then
        if command -v pjsub >/dev/null 2>&1; then
            pjsub "${SCRIPT_NAME}" "${TRAIN_ID}" "${INPUT_NAME}" "${BASE_CKPT}" "${TRAIN_MODE}"
            echo ">> [INFO] 已自动重提: pjsub ${SCRIPT_NAME} ${TRAIN_ID} ${INPUT_NAME} ${BASE_CKPT} ${TRAIN_MODE}"
        elif command -v qsub >/dev/null 2>&1; then
            qsub -v TRAIN_ID="${TRAIN_ID}",INPUT_NAME="${INPUT_NAME}",BASE_CKPT="${BASE_CKPT}",TRAIN_MODE="${TRAIN_MODE}",DP_TEST_N="${DP_TEST_N}" "${SCRIPT_NAME}"
            echo ">> [INFO] 已自动重提: qsub -v TRAIN_ID=${TRAIN_ID},INPUT_NAME=${INPUT_NAME},BASE_CKPT=${BASE_CKPT},TRAIN_MODE=${TRAIN_MODE},DP_TEST_N=${DP_TEST_N} ${SCRIPT_NAME}"
        else
            echo ">> [ERROR] 找不到 pjsub 或 qsub，无法自动重提"
            exit 1
        fi
    else
        echo ">> [ERROR] 找不到脚本: ${SCRIPT_NAME}"
        exit 1
    fi
fi
