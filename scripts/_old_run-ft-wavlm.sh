#!/bin/bash

set -e  # Exit on error
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1  # Set GPU device

# Seoul is fine, Singapore is shit
USE_DEVICE=0
OUTPUT_DIR=./exp/wavlm-ft-libri_2
DATA_DIR=/home/25/minh/ctc-assisted-asr-llm/data/LibriSpeech
MODEL_NAME="microsoft/wavlm-large"
CHECKPOINT=

# Define log paths
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/finetune.log"

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

rotate_file "$log_file"

CUDA_VISIBLE_DEVICES=$USE_DEVICE python src/finetune/_old_wavlm_ft_libri.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --learning_rate 3e-5 \
    --batch_size 8 \
    --accumulation_steps 64 \
    ${CHECKPOINT:+--checkpoint $CHECKPOINT}
