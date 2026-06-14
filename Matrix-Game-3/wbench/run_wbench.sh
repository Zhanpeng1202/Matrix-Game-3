#!/usr/bin/env bash
# End-to-end: prepare WBench navigation action inputs, then render with
# Matrix-Game-3 (action-controlled, causal). Generation only -- no scoring.
#
# Usage:
#   bash wbench/run_wbench.sh [CKPT_DIR] [NPROC]
# e.g. full pure-navigation split on 8 GPUs:
#   bash wbench/run_wbench.sh Matrix-Game-3.0 8
#
# Smoke test (2 cases) before the full run:
#   LIMIT=2 bash wbench/run_wbench.sh Matrix-Game-3.0 8
# Specific cases:
#   CASE_IDS=1,2,3 bash wbench/run_wbench.sh Matrix-Game-3.0 8
set -euo pipefail

CKPT_DIR="${1:-Matrix-Game-3.0}"
NPROC="${2:-8}"

SIZE="${SIZE:-704*1280}"
SELECTION="${SELECTION:-pure_nav}"
ULYSSES="${ULYSSES:-$NPROC}"          # sequence-parallel size; set ULYSSES=1 to disable SP
STEPS="${STEPS:-3}"                    # distilled few-step model
SEED="${SEED:-42}"
FPS="${FPS:-16}"
LIMIT="${LIMIT:-}"                     # optional: cap number of cases (smoke test)
CASE_IDS="${CASE_IDS:-}"               # optional: comma-separated case ids

WBENCH_ROOT="${WBENCH_ROOT:-/home/builder/workspace/WBench}"
WORK_DIR="${WORK_DIR:-${WBENCH_ROOT}/work_dirs/matrix_game_3}"
MANIFEST="${WORK_DIR}/manifest.json"
SAVE_DIR="${WORK_DIR}/videos"

cd "$(dirname "$0")/.."  # MG3 repo root

# If you hit a cuBLAS CUBLAS_STATUS_INVALID_VALUE error on this host, the system
# CUDA toolkit can shadow torch's bundled cuBLAS. Uncomment to drop the toolkit path:
# export LD_LIBRARY_PATH=/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

PREP_ARGS=(--wbench_root "${WBENCH_ROOT}" --work_dir "${WORK_DIR}" --selection "${SELECTION}")
[ -n "${LIMIT}" ] && PREP_ARGS+=(--limit "${LIMIT}")
[ -n "${CASE_IDS}" ] && PREP_ARGS+=(--case_ids "${CASE_IDS}")

echo "=== [1/2] Preparing action inputs (CPU) ==="
python wbench/prepare_wbench.py "${PREP_ARGS[@]}"

echo "=== [2/2] Generating videos (${NPROC} GPU, ulysses=${ULYSSES}) ==="
GEN_ARGS=(
    --ckpt_dir "${CKPT_DIR}"
    --manifest "${MANIFEST}"
    --save_dir "${SAVE_DIR}"
    --size "${SIZE}"
    --ulysses_size "${ULYSSES}"
    --num_inference_steps "${STEPS}"
    --seed "${SEED}"
    --fps "${FPS}"
    --use_int8
    --vae_type mg_lightvae
    --lightvae_pruning_rate 0.5
    --compile_vae
    --fa_version 3
    --resume
)
# FSDP is only used with sequence parallel (ulysses > 1).
if [ "${ULYSSES}" -gt 1 ]; then
    GEN_ARGS+=(--dit_fsdp --t5_fsdp)
fi

torchrun --nproc_per_node="${NPROC}" wbench/generate_wbench.py "${GEN_ARGS[@]}"

echo "Done. Videos -> ${SAVE_DIR}"
