#!/usr/bin/env bash
# LoRA rank sweep on EleutherAI/arithmetic only.
# Usage: bash run_sweep_eleuther.sh

set -e

DATASET="eleuther_arithmetic"
RANKS=(1 2 4 8 16 32)
STEPS=30000
BATCH_SIZE=16
LR=2e-5
MAX_LEN=128
SEED=0
PRECISION="fp16"
LOG_EVERY=500
METRIC_EVERY=500
SAVE_EVERY=6000
OUTPUT_BASE="output/sweep"

for RANK in "${RANKS[@]}"; do
    echo "========================================"
    echo "dataset=${DATASET} | rank=${RANK}"
    echo "========================================"

    python model_lora2.py \
        --dataset        ${DATASET} \
        --rank           ${RANK} \
        --steps          ${STEPS} \
        --batch_size     ${BATCH_SIZE} \
        --lr             ${LR} \
        --max_len        ${MAX_LEN} \
        --seed           ${SEED} \
        --precision      ${PRECISION} \
        --log_every      ${LOG_EVERY} \
        --metric_every   ${METRIC_EVERY} \
        --save_every     ${SAVE_EVERY} \
        --output_dir     ${OUTPUT_BASE}

    echo "Done dataset=${DATASET} rank=${RANK}"
    echo ""
done

echo "========================================"
echo "Eleuther sweep complete. Checkpoints in ${OUTPUT_BASE}/"
echo "========================================"
