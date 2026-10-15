#!/bin/bash
# =============================================================================
# Run ALL inference modes (nobias, gt, filter-<method>).
#
# Defaults target Japanese (CSJ eval1/2/3, CER, "ja" reading-based filter).
# Set LANGUAGE=en to get the LibriSpeech defaults back.
#
# Prerequisites:
#   1. bash decode-ctc.sh      → produces <CTC_DIR>/<split>_ctc.tsv
#   2. bash make-bias-list.sh  → produces <HOTWORDS_DIR>/<split>_bias_list.json
#
# Usage:
#   bash run-all-infer.sh                           # Japanese: nobias, gt, filter-ja
#
#   # Run only specific modes
#   INFER_MODES="nobias filter" bash run-all-infer.sh
#
#   # Custom settings
#   SPLITS="eval1_whisper_like_clean" bash run-all-infer.sh
#   EXP_DIR=exp/my_experiment bash run-all-infer.sh
#
#   # English (LibriSpeech, WER, F1/F2/F3 filters)
#   LANGUAGE=en bash run-all-infer.sh
#   LANGUAGE=en FILTER_METHODS="F3" bash run-all-infer.sh
# =============================================================================

set -e
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1
# CUDA order is FASTEST_FIRST: 0 = RTX 4000 Ada (20 GB), 1 = RTX 3090
USE_DEVICE="${USE_DEVICE:-0}"

ROOT_DIR=/home/25/minh/ctc-assisted-asr-llm
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
# Prompt/filter language: "ja" (CER) or "en" (WER)
LANGUAGE="${LANGUAGE:-ja}"

if [ "$LANGUAGE" = "ja" ]; then
    METRIC=cer
    DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/csj}"
    EXP_DIR="${EXP_DIR:-${ROOT_DIR}/exp/whisper-ft-ctc_llm_contextual}"
    INFER_CONFIG="${INFER_CONFIG:-${ROOT_DIR}/configs/inference_ja.yaml}"
    SPLITS="${SPLITS:-eval1_whisper_like_clean eval2_whisper_like_clean eval3_whisper_like_clean}"
    HOTWORDS_DIR="${HOTWORDS_DIR:-${ROOT_DIR}/data/lexicon/csj/hotwords}"
    CTC_DIR="${CTC_DIR:-${ROOT_DIR}/exp/whisper-ft-ctc_ja/ctc_decode}"
    COMMON_WORDS_PATH="${COMMON_WORDS_PATH:-${ROOT_DIR}/data/lexicon/csj/common_words_5k.txt}"
    FILTER_METHODS="${FILTER_METHODS:-ja}"
    ENCODER_TYPE="${ENCODER_TYPE:-whisper}"
else
    METRIC=wer
    DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/LibriSpeech}"
    # EXP_DIR="${EXP_DIR:-${ROOT_DIR}/exp/pt-wavlm_llm_libri_no-contextual}"
    EXP_DIR="${EXP_DIR:-${ROOT_DIR}/exp/wavlm-ft-ctc_llm_libri_no-contextual}"
    # EXP_DIR="${EXP_DIR:-${ROOT_DIR}/exp/wavlm-ft-ctc_llm_libri_contextual}"
    INFER_CONFIG="${INFER_CONFIG:-${ROOT_DIR}/configs/inference.yaml}"
    SPLITS="${SPLITS:-test-clean test-other}"
    HOTWORDS_DIR="${HOTWORDS_DIR:-${ROOT_DIR}/data/lexicon/LibriSpeech/bias_list_100}"
    CTC_DIR="${CTC_DIR:-/home/minh/ctc-assisted-asr-llm/exp/wavlm-ft-ctc_libri/ctc_decode}"
    COMMON_WORDS_PATH="${COMMON_WORDS_PATH:-${ROOT_DIR}/data/lexicon/LibriSpeech/common_words_5k.txt}"
    FILTER_METHODS="${FILTER_METHODS:-F1 F2 F3}"
    ENCODER_TYPE="${ENCODER_TYPE:-auto}"
fi

# Dataset layout: auto (csj if <split>/wav.scp exists), librispeech, csj
DATASET="${DATASET:-auto}"

# Experiment directory containing checkpoints/ and training_config.yaml
CHECKPOINT="${CHECKPOINT:-${EXP_DIR}/checkpoints/best_${METRIC}_model.pt}"
CONFIG="${CONFIG:-${EXP_DIR}/training_config.yaml}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DATA_DEVICE="${DATA_DEVICE:-cuda}"

# Filter parameters
PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-}"   # empty = use method default in INFER_CONFIG
WORD_NUM="${WORD_NUM:-}"                               # empty = use method default in INFER_CONFIG

# Inference modes (default: run everything)
INFER_MODES="${INFER_MODES:-nobias gt filter}"

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
echo "Checkpoint:      $CHECKPOINT"
echo "Language:        $LANGUAGE (${METRIC^^})"
echo "Data dir:        $DATA_DIR ($DATASET)"
echo "Infer config:    $INFER_CONFIG"
echo "Encoder type:    $ENCODER_TYPE"
echo "Splits:          $SPLITS"
echo "Infer modes:     $INFER_MODES"
echo "Filter methods:  $FILTER_METHODS"
echo "Bias lists:      $BIAS_LIST_FILES"
echo "CTC files:       $CTC_FILES"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper function
#   run_infer <infer_type> [filter_method]
#   e.g. run_infer nobias
#        run_infer filter F3
# ---------------------------------------------------------------------------
run_infer() {
    local infer_type=$1
    local filter_method=${2:-}  # only used when infer_type=filter

    # Output directory: nobias/, gt/, filter-F1/, filter-F2/, filter-F3/
    local dir_suffix="${infer_type}"
    if [ -n "$filter_method" ]; then
        dir_suffix="${infer_type}-${filter_method}"
    fi
    local output_dir="${EXP_DIR}/inference/${dir_suffix}"
    mkdir -p "$output_dir"

    local num_workers=0
    if [ "$DATA_DEVICE" != "cuda" ]; then
        num_workers="${NUM_WORKERS:-4}"
    fi

    echo ""
    echo "========================================"
    echo "  Running: infer_type=${infer_type}${filter_method:+, method=${filter_method}}"
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
        --dataset $DATASET \
        --splits $SPLITS \
        --batch-size $BATCH_SIZE \
        --num-beams $NUM_BEAMS \
        --max-new-tokens $MAX_NEW_TOKENS \
        --num-workers $num_workers \
        --data-device $DATA_DEVICE \
        --language $LANGUAGE \
        --encoder-type $ENCODER_TYPE"

    # Add bias list for gt / filter
    if [ "$infer_type" = "gt" ] || [ "$infer_type" = "filter" ]; then
        cmd="$cmd --bias-list-json $BIAS_LIST_FILES"
    fi

    # Add CTC file + filter params for filter
    if [ "$infer_type" = "filter" ]; then
        cmd="$cmd --ctc-file $CTC_FILES"
        cmd="$cmd --common-words-path $COMMON_WORDS_PATH"
        cmd="$cmd --filter-method $filter_method"
        [ -n "$PROBABILITY_THRESHOLD" ] && cmd="$cmd --probability-threshold $PROBABILITY_THRESHOLD"
        [ -n "$WORD_NUM" ] && cmd="$cmd --word-num $WORD_NUM"
    fi

    eval $cmd
    echo "  Done: ${dir_suffix}"
}

# ---------------------------------------------------------------------------
# Check data availability
# ---------------------------------------------------------------------------
missing_bias=0
for bf in $BIAS_LIST_FILES; do
    if [ ! -f "$bf" ]; then
        echo "WARNING: Bias list not found: $bf  → gt & filter modes will be skipped"
        echo "  Create with: bash make-bias-list.sh"
        missing_bias=1
        break
    fi
done

missing_ctc=0
for cf in $CTC_FILES; do
    if [ ! -f "$cf" ]; then
        echo "WARNING: CTC file not found: $cf  → filter mode will be skipped"
        echo "  Create with: bash decode-ctc.sh"
        missing_ctc=1
        break
    fi
done

# ---------------------------------------------------------------------------
# Run requested modes
# ---------------------------------------------------------------------------
for mode in $INFER_MODES; do
    case $mode in
        nobias)
            run_infer nobias
            ;;
        gt)
            if [ $missing_bias -eq 0 ]; then
                run_infer gt
            else
                echo "SKIP: gt (missing bias lists)"
            fi
            ;;
        filter)
            if [ $missing_bias -eq 0 ] && [ $missing_ctc -eq 0 ]; then
                for fm in $FILTER_METHODS; do
                    run_infer filter "$fm"
                done
            else
                echo "SKIP: filter (missing bias lists or CTC files)"
            fi
            ;;
        *)
            echo "WARNING: Unknown mode '$mode', skipping"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Cleanup & Summary
# ---------------------------------------------------------------------------
rm -rf pymp-* tmp*

# ---------------------------------------------------------------------------
# Build list of all result directories to check
# ---------------------------------------------------------------------------
ALL_RESULT_DIRS=""
for mode in $INFER_MODES; do
    if [ "$mode" = "filter" ]; then
        for fm in $FILTER_METHODS; do
            ALL_RESULT_DIRS="$ALL_RESULT_DIRS filter-${fm}"
        done
    else
        ALL_RESULT_DIRS="$ALL_RESULT_DIRS $mode"
    fi
done

echo ""
echo "================================================================"
echo "  ALL INFERENCE COMPLETED"
echo "================================================================"
echo "Results:"
for dir_suffix in $ALL_RESULT_DIRS; do
    result_dir="${EXP_DIR}/inference/${dir_suffix}"
    summary="${result_dir}/summary.json"
    if [ -f "$summary" ]; then
        echo "  ${dir_suffix}:  $(cat "$summary" | python3 -c "
import json, sys
d = json.load(sys.stdin)
wers = d.get('wer', {})
metric = d.get('metric', 'wer').upper()
parts = [f'{k}={v*100:.2f}%' for k,v in wers.items()]
print(metric + ': ' + ', '.join(parts))
" 2>/dev/null || echo "see ${summary}")"
    else
        echo "  ${dir_suffix}:  (skipped)"
    fi
done
echo "================================================================"
