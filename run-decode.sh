#!/bin/bash

# Script to run inference on test set with fine-tuned WavLM model

# Configuration
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
DATA_DIR=${DATA_DIR:-"/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech"}
EXP_OUTPUT_DIR="exp/wavlm-ft-libri_2"
VOCAB_PATH=${VOCAB_PATH:-"$EXP_OUTPUT_DIR/vocab.json"}

# Fine-tuned model
MODEL_PATH=${MODEL_PATH:-"$EXP_OUTPUT_DIR/final"}
OUTPUT_DIR=${OUTPUT_DIR:-"$EXP_OUTPUT_DIR/inferences/finetuned"}

# Paper model
# MODEL_PATH="/home/25/minh/CTC-assisted-ASR-LLM/models/wavlm_large_ft_libri960_char.pt"
# OUTPUT_DIR=${OUTPUT_DIR:-"$EXP_OUTPUT_DIR/inferences/org_ft"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-16}
DEVICE=${DEVICE:-"cuda"}
export CUDA_VISIBLE_DEVICES=0,1  # Set GPU device
USE_DEVICE=0,1

if [ ! -f "$VOCAB_PATH" ]; then
    echo "Error: Vocab path does not exist: $VOCAB_PATH"
    exit 1
fi

# Run inference
CUDA_VISIBLE_DEVICES=$USE_DEVICE python src/decode_encoder.py \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --vocab_path "$VOCAB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE"
