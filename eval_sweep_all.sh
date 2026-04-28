#!/usr/bin/env bash
# Evaluate all rank-sweep checkpoints on their respective test sets.
# Writes per-run JSON to {run_dir}/eval_result.json
# Usage: bash eval_sweep_all.sh

OUTPUT_BASE="output/sweep"
RANKS=(1 2 4 8 16 32)
MAX_EXAMPLES=2000

# ── Baseline (no LoRA) ──────────────────────────────────────────
for DATASET in arithmetic eleuther_arithmetic; do
    BASELINE_DIR="${OUTPUT_BASE}/baseline_${DATASET}"
    mkdir -p ${BASELINE_DIR}
    if [ -f "${BASELINE_DIR}/eval_result.json" ]; then
        echo "Skipping baseline ${DATASET} — already done"
    else
        echo "========================================"
        echo "Baseline (no LoRA) | dataset=${DATASET}"
        echo "========================================"
        python3 eval_lora.py \
            --baseline \
            --dataset           ${DATASET} \
            --max_examples      ${MAX_EXAMPLES} \
            --metric_diffusion_steps 64 \
            --metric_max_new_tokens  64 \
            --output_json       ${BASELINE_DIR}/eval_result.json
        echo "Done baseline ${DATASET}"
        echo ""
    fi
done

# ── LoRA ranks ───────────────────────────────────────────────────
for DATASET in arithmetic eleuther_arithmetic; do
    for RANK in "${RANKS[@]}"; do
        RUN_DIR="${OUTPUT_BASE}/${DATASET}-r${RANK}-lr2e-05-bs16"

        # Try checkpoints from highest step to lowest, skip corrupted ones
        CKPT=""
        for CANDIDATE in $(ls ${RUN_DIR}/lora_step*.pt 2>/dev/null | sort -V -r); do
            if python3 -c "import torch; torch.load('${CANDIDATE}', map_location='cpu')" 2>/dev/null; then
                CKPT="${CANDIDATE}"
                break
            else
                echo "WARNING: ${CANDIDATE} is corrupted, trying previous checkpoint"
            fi
        done

        if [ -z "$CKPT" ]; then
            echo "WARNING: no valid checkpoint found in ${RUN_DIR}, skipping"
            continue
        fi

        if [ -f "${RUN_DIR}/eval_result.json" ]; then
            echo "Skipping dataset=${DATASET} rank=${RANK} — eval_result.json already exists"
            continue
        fi

        echo "========================================"
        echo "Evaluating dataset=${DATASET} rank=${RANK}"
        echo "Checkpoint: ${CKPT}"
        echo "========================================"

        python3 eval_lora.py \
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

echo "######################################## Plotting"

python3 plot_sweep.py \
    --output_dir ${OUTPUT_BASE} \
    --save_dir   ${OUTPUT_BASE}

echo "Plot saved to ${OUTPUT_BASE}/sweep_accuracy.png"
