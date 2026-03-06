#!/bin/bash
# =============================================================================
# Run ALL three inference modes (nobias, gt, filter) sequentially.
#
# Prerequisites:
#   1. bash decode-ctc.sh      → produces data/ctc_decode/*_ctc.tsv
#   2. bash make-bias-list.sh  → produces data/hotwords/*_bias_list.json
#
# Usage:
#   bash run-all-infer.sh
#
#   # Custom settings
#   SPLITS="test-clean" bash run-all-infer.sh
#   EXP_DIR=exp/my_experiment bash run-all-infer.sh
# =============================================================================

set -e
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1
USE_DEVICE=0

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/home/25/minh/ctc-assisted-asr-llm/data/LibriSpeech}"
EXP_DIR="${EXP_DIR:-/home/25/minh/ctc-assisted-asr-llm/exp/wavlm-ft-ctc_llm_libri_contextual}"
CHECKPOINT="${CHECKPOINT:-${EXP_DIR}/checkpoints/best_wer_model.pt}"
CONFIG="${CONFIG:-${EXP_DIR}/training_config.yaml}"
INFER_CONFIG="${INFER_CONFIG:-config/inference.yaml}"
SPLITS="${SPLITS:-test-clean test-other}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DATA_DEVICE="${DATA_DEVICE:-cuda}"

# Bias list / CTC directories
HOTWORDS_DIR="${HOTWORDS_DIR:-/home/25/minh/ctc-assisted-asr-llm/data/lexicon/LibriSpeech/hotwords}"
CTC_DIR="${CTC_DIR:-/home/25/minh/ctc-assisted-asr-llm/exp/wavlm-ft-libri_2/ctc_decode}"

# Filter parameters
COMMON_WORDS_PATH="${COMMON_WORDS_PATH:-/home/25/minh/ctc-assisted-asr-llm/data/lexicon/LibriSpeech/common_words_5k.txt}"
PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-0.9}"
WORD_NUM="${WORD_NUM:-15}"

# ---------------------------------------------------------------------------
# Auto-build space-separated file lists from SPLITS
# ---------------------------------------------------------------------------
BIAS_LIST_FILES=""
CTC_FILES=""
for split in $SPLITS; do
    bf="${HOTWORDS_DIR}/${split}_bias_list.json"
    cf="${CTC_DIR}/${split}_ctc.tsv"
    BIAS_LIST_FILES="${BIAS_LIST_FILES:+$BIAS_LIST_FILES }$bf"
    CTC_FILES="${CTC_FILES:+$CTC_FILES }$cf"
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "================================================================"
echo "  CTC-assisted LLM ASR – Run All Inference Modes"
echo "================================================================"
echo "Checkpoint:   $CHECKPOINT"
echo "Splits:       $SPLITS"
echo "Bias lists:   $BIAS_LIST_FILES"
echo "CTC files:    $CTC_FILES"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------
run_infer() {
    local infer_type=$1
    local output_dir="${EXP_DIR}/inference/${infer_type}"
    mkdir -p "$output_dir"

    local num_workers=0
    if [ "$DATA_DEVICE" != "cuda" ]; then
        num_workers="${NUM_WORKERS:-4}"
    fi

    echo ""
    echo "========================================"
    echo "  Running: infer_type=${infer_type}"
    echo "  Output:  ${output_dir}"
    echo "========================================"

    local cmd="CUDA_VISIBLE_DEVICES=$USE_DEVICE python3 src/inference/inference.py \
        --mode eval \
        --infer-type $infer_type \
        --checkpoint $CHECKPOINT \
        --config $CONFIG \
        --infer-config $INFER_CONFIG \
        --output-dir $output_dir \
        --data-dir $DATA_DIR \
        --splits $SPLITS \
        --batch-size $BATCH_SIZE \
        --num-beams $NUM_BEAMS \
        --max-new-tokens $MAX_NEW_TOKENS \
        --num-workers $num_workers \
        --data-device $DATA_DEVICE"

    # Add bias list for gt / filter
    if [ "$infer_type" = "gt" ] || [ "$infer_type" = "filter" ]; then
        cmd="$cmd --bias-list-json $BIAS_LIST_FILES"
    fi

    # Add CTC file + filter params for filter
    if [ "$infer_type" = "filter" ]; then
        cmd="$cmd --ctc-file $CTC_FILES"
        cmd="$cmd --common-words-path $COMMON_WORDS_PATH"
        cmd="$cmd --probability-threshold $PROBABILITY_THRESHOLD"
        cmd="$cmd --word-num $WORD_NUM"
    fi

    eval $cmd
    echo "  Done: ${infer_type}"
}

# ---------------------------------------------------------------------------
# 1. nobias
# ---------------------------------------------------------------------------
# run_infer nobias

# ---------------------------------------------------------------------------
# 2. gt  (requires bias lists)
# ---------------------------------------------------------------------------
missing_bias=0
for bf in $BIAS_LIST_FILES; do
    if [ ! -f "$bf" ]; then
        echo "WARNING: Bias list not found: $bf  → skipping gt & filter modes"
        echo "  Create with: bash make-bias-list.sh"
        missing_bias=1
        break
    fi
done

if [ $missing_bias -eq 0 ]; then
    run_infer gt
fi

# ---------------------------------------------------------------------------
# 3. filter  (requires bias lists + CTC files)
# ---------------------------------------------------------------------------
missing_ctc=0
for cf in $CTC_FILES; do
    if [ ! -f "$cf" ]; then
        echo "WARNING: CTC file not found: $cf  → skipping filter mode"
        echo "  Create with: bash decode-ctc.sh"
        missing_ctc=1
        break
    fi
done

if [ $missing_bias -eq 0 ] && [ $missing_ctc -eq 0 ]; then
    run_infer filter
fi

# ---------------------------------------------------------------------------
# Cleanup & Summary
# ---------------------------------------------------------------------------
rm -rf pymp-* tmp*

echo ""
echo "================================================================"
echo "  ALL INFERENCE COMPLETED"
echo "================================================================"
echo "Results:"
for mode in nobias gt filter; do
    result_dir="${EXP_DIR}/inference/${mode}"
    summary="${result_dir}/summary.json"
    if [ -f "$summary" ]; then
        echo "  ${mode}:  $(cat "$summary" | python3 -c "
import json, sys
d = json.load(sys.stdin)
wers = d.get('wer', {})
parts = [f'{k}={v*100:.2f}%' for k,v in wers.items()]
print(', '.join(parts))
" 2>/dev/null || echo "see ${summary}")"
    else
        echo "  ${mode}:  (skipped)"
    fi
done
echo "================================================================"
