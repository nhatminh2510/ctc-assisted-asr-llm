#!/bin/bash
# =============================================================================
# Inference script for CTC-assisted LLM-based Contextual ASR
#
# Three inference modes:
#   nobias  – No hotwords in prompt (baseline)
#   gt      – Ground-truth hotwords from bias list (oracle upper bound)
#   filter  – CTC-filtered hotwords from bias list (the paper's method)
#
# Usage:
#   # Default: nobias mode on test-clean and test-other
#   bash run-inference.sh
#
#   # GT mode (requires bias list)
#   INFER_TYPE=gt \
#       BIAS_LIST_JSON="data/hotwords/test-clean_bias_list.json data/hotwords/test-other_bias_list.json" \
#       bash run-inference.sh
#
#   # Filter mode (requires bias list + CTC file)
#   INFER_TYPE=filter \
#       BIAS_LIST_JSON="data/hotwords/test-clean_bias_list.json data/hotwords/test-other_bias_list.json" \
#       CTC_FILE="data/ctc_decode/test-clean_ctc.tsv data/ctc_decode/test-other_ctc.tsv" \
#       bash run-inference.sh
#
#   # Japanese (CSJ): CER scoring, JA prompt/filter, Whisper encoder
#   DATA_DIR=data/csj SPLITS="eval1_whisper_like_clean" LANGUAGE=ja \
#       ENCODER_TYPE=whisper INFER_CONFIG=configs/inference_ja.yaml \
#       EXP_DIR=exp/whisper_llm_ja bash run-inference.sh
#
#   # Transcribe a single audio file
#   MODE=transcribe AUDIO=/path/to/audio.flac bash run-inference.sh
# =============================================================================

set -e
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1
USE_DEVICE=0

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
ROOT_DIR=/home/25/minh/ctc-assisted-asr-llm
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/LibriSpeech}"
# Dataset layout: auto (csj if <split>/wav.scp exists), librispeech, csj
DATASET="${DATASET:-auto}"

# Experiment directory containing checkpoints/ and training_config.yaml
EXP_DIR="${EXP_DIR:-${ROOT_DIR}/exp/wavlm-ft-ctc_llm_libri_no-contextual}"

# Prompt/filter language: "en" (WER) or "ja" (CER)
LANGUAGE="${LANGUAGE:-en}"
if [ "$LANGUAGE" = "ja" ]; then METRIC=cer; else METRIC=wer; fi

# Checkpoint – defaults to best_wer_model.pt (en) / best_cer_model.pt (ja)
CHECKPOINT="${CHECKPOINT:-${EXP_DIR}/checkpoints/best_${METRIC}_model.pt}"

# Config file – defaults to training_config.yaml in the experiment
CONFIG="${CONFIG:-${EXP_DIR}/training_config.yaml}"
INFER_CONFIG="${INFER_CONFIG:-${ROOT_DIR}/configs/inference.yaml}"

# Mode: "eval" (dataset evaluation) or "transcribe" (single file)
MODE="${MODE:-eval}"

# Inference type: "nobias", "gt", or "filter"
INFER_TYPE="${INFER_TYPE:-nobias}"

# Speech encoder family: "auto", "wavlm", "hubert", "wav2vec2", "whisper"
ENCODER_TYPE="${ENCODER_TYPE:-auto}"

# Output directory for results (auto: <exp-dir>/inference/<infer_type>/)
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_DIR}/inference/${INFER_TYPE}}"

# Dataset splits to evaluate (space-separated)
SPLITS="${SPLITS:-test-clean test-other}"

# Audio file path (only for transcribe mode)
AUDIO="${AUDIO:-}"

# Generation parameters
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DATA_DEVICE="${DATA_DEVICE:-cuda}"
if [ "$DATA_DEVICE" == "cuda" ]; then
    NUM_WORKERS=0  
else
    NUM_WORKERS="${NUM_WORKERS:-4}"
fi

# --- Bias list / CTC file (for gt and filter modes) ---
# Space-separated list of files (one per split)
BIAS_LIST_JSON="${BIAS_LIST_JSON:-}"
CTC_FILE="${CTC_FILE:-}"

# --- Filter parameters ---
# Method: F1/F2/F3 (English) or ja (Japanese reading-based filter)
FILTER_METHOD="${FILTER_METHOD:-F3}"
COMMON_WORDS_PATH="${COMMON_WORDS_PATH:-/home/25/minh/ctc-assisted-asr-llm/data/lexicon/LibriSpeech/common_words_5k.txt}"
# The ja filter scores on a different scale (default 0.5 in inference_ja.yaml),
# so for it an unset value means "take it from INFER_CONFIG", not 0.9.
if [ "$FILTER_METHOD" = "ja" ]; then
    PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-}"
    WORD_NUM="${WORD_NUM:-}"
else
    PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-0.9}"
    WORD_NUM="${WORD_NUM:-15}"
fi

# --- Hotwords for transcribe mode ---
HOTWORDS="${HOTWORDS:-}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints in ${EXP_DIR}/checkpoints/:"
    ls -1 "${EXP_DIR}/checkpoints/" 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

if [ "$MODE" = "eval" ] && [ ! -d "$DATA_DIR" ] && [ ! -L "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ "$MODE" = "transcribe" ] && [ -z "$AUDIO" ]; then
    echo "ERROR: AUDIO path is required in transcribe mode"
    echo "Usage: MODE=transcribe AUDIO=/path/to/file.flac bash run-inference.sh"
    exit 1
fi

if [ "$INFER_TYPE" = "gt" ] || [ "$INFER_TYPE" = "filter" ]; then
    if [ -z "$BIAS_LIST_JSON" ]; then
        echo "ERROR: BIAS_LIST_JSON is required for infer_type=$INFER_TYPE"
        echo "  Create it with: bash make-bias-list.sh"
        echo "  Example: BIAS_LIST_JSON=\"data/hotwords/test-clean_bias_list.json data/hotwords/test-other_bias_list.json\""
        exit 1
    fi
    for bf in $BIAS_LIST_JSON; do
        if [ ! -f "$bf" ]; then
            echo "ERROR: Bias list file not found: $bf"
            exit 1
        fi
    done
fi

if [ "$INFER_TYPE" = "filter" ]; then
    if [ -z "$CTC_FILE" ]; then
        echo "ERROR: CTC_FILE is required for infer_type=filter"
        echo "  Create it with: bash decode-ctc.sh"
        echo "  Example: CTC_FILE=\"data/ctc_decode/test-clean_ctc.tsv data/ctc_decode/test-other_ctc.tsv\""
        exit 1
    fi
    for cf in $CTC_FILE; do
        if [ ! -f "$cf" ]; then
            echo "ERROR: CTC file not found: $cf"
            exit 1
        fi
    done
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

CMD="CUDA_VISIBLE_DEVICES=$USE_DEVICE python3 src/inference/inference.py \
    --mode $MODE \
    --infer-type $INFER_TYPE \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --infer-config $INFER_CONFIG \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --num-beams $NUM_BEAMS \
    --max-new-tokens $MAX_NEW_TOKENS \
    --num-workers $NUM_WORKERS \
    --language $LANGUAGE \
    --encoder-type $ENCODER_TYPE \
    --data-device $DATA_DEVICE"
    

if [ "$MODE" = "eval" ]; then
    CMD="$CMD --data-dir $DATA_DIR --dataset $DATASET --splits $SPLITS"
fi

if [ "$MODE" = "transcribe" ] && [ -n "$AUDIO" ]; then
    CMD="$CMD --audio $AUDIO"
fi

if [ -n "$HOTWORDS" ]; then
    CMD="$CMD --hotwords $HOTWORDS"
fi

if [ -n "$BIAS_LIST_JSON" ]; then
    CMD="$CMD --bias-list-json $BIAS_LIST_JSON"
fi

if [ -n "$CTC_FILE" ]; then
    CMD="$CMD --ctc-file $CTC_FILE"
fi

if [ "$INFER_TYPE" = "filter" ]; then
    CMD="$CMD --filter-method $FILTER_METHOD"
    CMD="$CMD --common-words-path $COMMON_WORDS_PATH"
    [ -n "$PROBABILITY_THRESHOLD" ] && CMD="$CMD --probability-threshold $PROBABILITY_THRESHOLD"
    [ -n "$WORD_NUM" ] && CMD="$CMD --word-num $WORD_NUM"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================"
echo "CTC-assisted LLM ASR – Inference"
echo "========================================"
echo "Mode:           $MODE"
echo "Infer type:     $INFER_TYPE"
echo "Language:       $LANGUAGE"
echo "Encoder type:   $ENCODER_TYPE"
echo "Checkpoint:     $CHECKPOINT"
echo "Output:         $OUTPUT_DIR"
if [ "$MODE" = "eval" ]; then
    echo "Data dir:       $DATA_DIR ($DATASET)"
    echo "Splits:         $SPLITS"
fi
if [ "$MODE" = "transcribe" ]; then
    echo "Audio:          $AUDIO"
fi
echo "Batch size:     $BATCH_SIZE"
echo "Num beams:      $NUM_BEAMS"
if [ "$INFER_TYPE" = "gt" ] || [ "$INFER_TYPE" = "filter" ]; then
    echo "Bias list:      $BIAS_LIST_JSON"
fi
if [ "$INFER_TYPE" = "filter" ]; then
    echo "Filter method:  $FILTER_METHOD"
    echo "CTC file:       $CTC_FILE"
    echo "Threshold:      ${PROBABILITY_THRESHOLD:-<from INFER_CONFIG>}"
    echo "Word num:       ${WORD_NUM:-<from INFER_CONFIG>}"
    echo "Common words:   $COMMON_WORDS_PATH"
fi
echo "========================================"
echo ""

eval $CMD

# Cleanup temp dirs
rm -rf pymp-* tmp*

echo ""
echo "========================================"
echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "========================================"

