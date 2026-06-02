#!/bin/bash
#PJM -L rscgrp=a-pj24001724
#PJM -L node=1
#PJM --mpi proc=120
#PJM -L elapse=128:00:00
#PJM -j
set -eu

module load intel
module load impi
#module load vasp
source /home/pj24001724/ku40000345/wu/deepmd_kit/use.sh

ulimit -s unlimited
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Edit these values before submitting/running on Linux.
# If you need to override the activated environment Python, set:
# PYTHON_BIN=/path/to/python ./wu_bondcount.sh
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SEARCH_PY="${SCRIPT_DIR}/search.py"

INPUT_ROOT="${INPUT_ROOT:-.}"
STRUCTURE_DIR="${STRUCTURE_DIR:-}"
PATTERNS="${PATTERNS:-*.cif}"
EXCLUDE_DIRS="${EXCLUDE_DIRS:-interface_bond_cifs}"

OUTPUT_DIR="${OUTPUT_DIR:-interface_bond_cifs}"
SUMMARY="${SUMMARY:-interface_bond_summary.csv}"
PROGRESS_MARKDOWN="${PROGRESS_MARKDOWN:-interface_bond_progress.md}"
WORKERS="${WORKERS:-90}"

MIN_INTERFACE_BONDS="${MIN_INTERFACE_BONDS:-2}"
MOLECULE_SEED_SYMBOLS="${MOLECULE_SEED_SYMBOLS:-C,H}"
MOLECULE_SYMBOLS="${MOLECULE_SYMBOLS:-C,H,O,N,S}"
MOLECULE_BOND_SYMBOLS="${MOLECULE_BOND_SYMBOLS:-C}"
SURFACE_SYMBOLS="${SURFACE_SYMBOLS:-Zn,O}"
INTERFACE_BOND_CUTOFF_SCALE="${INTERFACE_BOND_CUTOFF_SCALE:-1.25}"
INTERFACE_MIN_CUTOFF="${INTERFACE_MIN_CUTOFF:-0.7}"
INTERFACE_MAX_CUTOFF="${INTERFACE_MAX_CUTOFF:-2.4}"
DRY_RUN_REQUESTED=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN_REQUESTED=1
fi

set -- "${SEARCH_PY}" \
  --input-root "${INPUT_ROOT}" \
  --patterns "${PATTERNS}" \
  --exclude-dirs "${EXCLUDE_DIRS}" \
  --output-dir "${OUTPUT_DIR}" \
  --summary "${SUMMARY}" \
  --progress-markdown "${PROGRESS_MARKDOWN}" \
  --workers "${WORKERS}" \
  --min-interface-bonds "${MIN_INTERFACE_BONDS}" \
  --molecule-seed-symbols "${MOLECULE_SEED_SYMBOLS}" \
  --molecule-symbols "${MOLECULE_SYMBOLS}" \
  --molecule-bond-symbols "${MOLECULE_BOND_SYMBOLS}" \
  --surface-symbols "${SURFACE_SYMBOLS}" \
  --interface-bond-cutoff-scale "${INTERFACE_BOND_CUTOFF_SCALE}" \
  --interface-min-cutoff "${INTERFACE_MIN_CUTOFF}" \
  --interface-max-cutoff "${INTERFACE_MAX_CUTOFF}"

if [ -n "${STRUCTURE_DIR}" ]; then
  set -- "$@" --structure-dir "${STRUCTURE_DIR}"
fi

# Pass --dry-run to this shell script if you only want CSV/Markdown outputs.
if [ "${DRY_RUN_REQUESTED}" -eq 1 ]; then
  set -- "$@" --dry-run
fi

exec "${PYTHON_BIN}" "$@"
