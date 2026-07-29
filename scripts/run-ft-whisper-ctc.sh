#!/bin/bash
# =============================================================================
# Fine-tune a Whisper-encoder CTC model that emits kana (for the Japanese
# hotword filter). Run inside a container that ships a kana backend
# (fugashi/SudachiPy), e.g. /home/25/minh/Prefix-tree/prefix-tree.sif:
#
#   singularity exec --writable-tmpfs --nv \
#     --bind /home,/home-local,/CDshare,/kailab,/App,/LabWork,/UserTmp,/local,/Mirror \
#     --no-home /home/25/minh/Prefix-tree/prefix-tree.sif \
#     bash scripts/run-ft-whisper-ctc.sh
# =============================================================================

set -e
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1

USE_DEVICE=0
CONFIG=configs/tuning/whisper-ctc-ja.yaml
CHECKPOINT=                                   # set to resume, e.g. exp/whisper-ft-ctc_ja/checkpoint-8000
OUTPUT_DIR=/home/25/minh/ctc-assisted-asr-llm/exp/whisper-ft-ctc_ja

# ---- Optional CLI overrides (uncomment / edit as needed) ----
OVERRIDES=(
    # model.name=openai/whisper-small
    # optimizer.peak_lr=1e-4
    # training.batch_size=16
    output.output_dir=$OUTPUT_DIR
)

log_dir="$OUTPUT_DIR/logs"
mkdir -p "$log_dir"
rotate_file() {
    local target_file=$1
    local base_name=${target_file%.log}
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
CUDA_VISIBLE_DEVICES=$USE_DEVICE python src/finetune/whisper_ft_ctc.py \
    --config "$CONFIG" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${OVERRIDES[@]:+--override "${OVERRIDES[@]}"}
