#!/usr/bin/env bash
# Train all eleuther_arithmetic ranks, then eval all sweep checkpoints
# (both arithmetic and eleuther_arithmetic), then plot accuracy vs rank.
# Usage: bash run_eleuther_then_eval.sh

set -e

OUTPUT_BASE="output/sweep"
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
MAX_EVAL_EXAMPLES=500   # set to -1 for full test set

# ──────────────────────────────────────────────
# PHASE 1: Train eleuther_arithmetic ranks
# ──────────────────────────────────────────────
echo "######################################## PHASE 1: Training"

for RANK in "${RANKS[@]}"; do
    RUN_DIR="${OUTPUT_BASE}/eleuther_arithmetic-r${RANK}-lr${LR:?}-bs${BATCH_SIZE}"
    FINAL_CKPT="${OUTPUT_BASE}/eleuther_arithmetic-r${RANK}-lr2e-05-bs${BATCH_SIZE}/lora_step${STEPS}.pt"

    if [ -f "${FINAL_CKPT}" ]; then
        echo "Skipping eleuther_arithmetic rank=${RANK} — final checkpoint already exists"
        continue
    fi

    echo "========================================"
    echo "Training eleuther_arithmetic | rank=${RANK}"
    echo "========================================"

    python model_lora2.py \
        --dataset        eleuther_arithmetic \
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

    echo "Done eleuther_arithmetic rank=${RANK}"
    echo ""
done

echo "######################################## PHASE 1 complete"

# ──────────────────────────────────────────────
# PHASE 2: Eval all checkpoints
# ──────────────────────────────────────────────
echo "######################################## PHASE 2: Evaluation"

for DATASET in arithmetic eleuther_arithmetic; do
    for RANK in "${RANKS[@]}"; do
        RUN_DIR="${OUTPUT_BASE}/${DATASET}-r${RANK}-lr2e-05-bs16"

        CKPT=$(ls ${RUN_DIR}/lora_step*.pt 2>/dev/null | sort -V | tail -1)

        if [ -z "$CKPT" ]; then
            echo "WARNING: no checkpoint in ${RUN_DIR}, skipping"
            continue
        fi

        echo "========================================"
        echo "Evaluating dataset=${DATASET} rank=${RANK}"
        echo "Checkpoint: ${CKPT}"
        echo "========================================"

        python eval_lora.py \
            --checkpoint             ${CKPT} \
            --rank                   ${RANK} \
            --dataset                ${DATASET} \
            --max_examples           ${MAX_EVAL_EXAMPLES} \
            --metric_diffusion_steps 64 \
            --metric_max_new_tokens  64 \
            --output_json            ${RUN_DIR}/eval_result.json

        echo "Done dataset=${DATASET} rank=${RANK}"
        echo ""
    done
done

echo "######################################## PHASE 2 complete"

# ──────────────────────────────────────────────
# PHASE 3: Plot
# ──────────────────────────────────────────────
echo "######################################## PHASE 3: Plotting"

python plot_sweep.py \
    --output_dir ${OUTPUT_BASE} \
    --save_path  ${OUTPUT_BASE}/sweep_accuracy.png

echo "######################################## All done"
echo "Plot saved to ${OUTPUT_BASE}/sweep_accuracy.png"
