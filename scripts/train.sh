#!/bin/bash
# Training script for CTC-assisted LLM-based ASR

set -e  # Exit on error
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1  # Set GPU device
USE_DEVICE=0
# Seoul is fine, 
# Singapore, Toronto is shit

# Check if data exists
data_dir="/home/25/minh/ctc-assisted-asr-llm/data/LibriSpeech"
if [ ! -d "$data_dir" ] && [ ! -L "$data_dir" ]; then
    echo "ERROR: Data not found at $data_dir"
    echo "Please ensure data is properly placed in the data directory."
    exit 1
fi

# pt-wavlm_llm_no-contextual
conf="/home/minh/ctc-assisted-asr-llm/configs/train_pt-wavlm_llm_no-contextual.yaml"
# conf="/home/minh/ctc-assisted-asr-llm/configs/train_pt-wavlm_llm_no-contextual_GB10.yaml"
output_dir="/home/25/minh/ctc-assisted-asr-llm/exp/pt-wavlm_llm_libri_no-contextual"

# wavlm-ft-ctc_llm_no-contextual
# conf="/home/minh/ctc-assisted-asr-llm/configs/train_wavlm-ft-ctc_llm_no-contextual.yaml"
# output_dir="/home/25/minh/ctc-assisted-asr-llm/exp/wavlm-ft-ctc_llm_libri_no-contextual"

# wavlm-ft-ctc_llm_contextual
# conf="/home/minh/ctc-assisted-asr-llm/configs/train_wavlm-ft-ctc_llm_contextual.yaml"
# output_dir="/home/25/minh/ctc-assisted-asr-llm/exp/wavlm-ft-ctc_llm_libri_contextual"

# Check if config exists
if [ ! -f "$conf" ]; then
    echo "ERROR: Config file not found at $conf"
    exit 1
fi

# Create output directory if not exists
log_dir="$output_dir/logs"
mkdir -p $output_dir/checkpoints
mkdir -p "$log_dir"
echo "Starting training..."
echo "Logs will be saved to: $log_dir/training.log"
echo "Checkpoints will be saved to: $output_dir/checkpoints/"

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
rotate_file "$log_dir/eval.log"
# Load checkpoint
checkpoint=
if [ -n "$checkpoint" ]; then
    echo "Resuming training from checkpoint: $checkpoint"
fi

# Run training
# CUDA_VISIBLE_DEVICES=$USE_DEVICE python src/train_GB10.py \
CUDA_VISIBLE_DEVICES=$USE_DEVICE python src/train.py \
    --config $conf \
    --data-dir $data_dir \
    --output-dir $output_dir \
    ${checkpoint:+--load-checkpoint $checkpoint} 

rm -rf pymp* tmp*
echo ""
echo "========================================"
echo "Training completed!"
echo "Output saved to: $output_dir/"
echo "========================================"
