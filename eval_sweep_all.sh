#!/usr/bin/env bash
# Evaluate all rank-sweep checkpoints on their respective test sets.
# Writes per-run JSON to {run_dir}/eval_result.json
# Usage: bash eval_sweep_all.sh

set -e

OUTPUT_BASE="output/sweep"
RANKS=(1 2 4 8 16 32)
MAX_EXAMPLES=500   # set to -1 for full test set (slow)

for DATASET in arithmetic eleuther_arithmetic; do
    for RANK in "${RANKS[@]}"; do
        RUN_DIR="${OUTPUT_BASE}/${DATASET}-r${RANK}-lr2e-05-bs16"

        # Find the checkpoint with the highest step number
        CKPT=$(ls ${RUN_DIR}/lora_step*.pt 2>/dev/null | sort -V | tail -1)

        if [ -z "$CKPT" ]; then
            echo "WARNING: no checkpoint found in ${RUN_DIR}, skipping"
            continue
        fi

        echo "========================================"
        echo "Evaluating dataset=${DATASET} rank=${RANK}"
        echo "Checkpoint: ${CKPT}"
        echo "========================================"

        python eval_lora.py \
            --checkpoint        ${CKPT} \
            --rank              ${RANK} \
            --dataset           ${DATASET} \
            --max_examples      ${MAX_EXAMPLES} \
            --metric_diffusion_steps 64 \
            --metric_max_new_tokens  64 \
            --output_json       ${RUN_DIR}/eval_result.json

        echo "Done dataset=${DATASET} rank=${RANK}"
        echo ""
    done
done

echo "========================================"
echo "All evaluations complete."
echo "Results in ${OUTPUT_BASE}/*/eval_result.json"
echo "========================================"
