#!/bin/bash

set -e  # Exit on error
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1

USE_DEVICE=0
CONFIG=configs/tuning/wavlm-ctc.yaml
CHECKPOINT=/home/minh/ctc-assisted-asr-llm/exp/wavlm-ft-ctc_libri_new/checkpoint-10000
OUTPUT_DIR=/home/minh/ctc-assisted-asr-llm/exp/wavlm-ft-ctc_libri_new

# ---- Optional CLI overrides (uncomment / edit as needed) ----
OVERRIDES=(
    # optimizer.peak_lr=1e-4
    # training.batch_size=16
    # training.gradient_accumulation_steps=8
    output.output_dir=$OUTPUT_DIR
)

log_dir="$OUTPUT_DIR/logs"
mkdir -p "$log_dir"
rotate_file() {
    local target_file=$1
    local base_name=${target_file%.log} # Removes .log extension for the counter

    if [ -f "$target_file" ]; then
        counter=1
        while [ -f "${base_name}.${counter}.log" ]; do
            ((counter++))
        done
        mv "$target_file" "${base_name}.${counter}.log"
        echo "Rotated: $target_file -> ${base_name}.${counter}.log"
    fi
}

rotate_file "$log_dir/training.log"

# ---- Run ----
CUDA_VISIBLE_DEVICES=$USE_DEVICE python src/finetune/wavlm_ft_ctc.py \
    --config "$CONFIG" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${OVERRIDES[@]:+--override "${OVERRIDES[@]}"}
