#!/bin/bash
# =============================================================================
# Decode LibriSpeech test sets using finetuned WavLM-CTC model.
#
# Produces CTC pseudo-label files used by the "filter" inference mode.
#
# Usage:
#   bash decode-ctc.sh
#
#   # Custom model / splits
#   WAVLM_MODEL=exp/wavlm-ft-ctc_libri_new-libri/final SPLITS="test-clean" bash decode-ctc.sh
# =============================================================================

set -e
export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1
USE_DEVICE=0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/home/25/minh/ctc-assisted-asr-llm/data/LibriSpeech}"

# Default to the WavLM-CTC model from the current experiment
EXP_DIR="${EXP_DIR:-exp/wavlm-ft-ctc_libri_new}"

# WAVLM_MODEL="${WAVLM_MODEL:-${EXP_DIR}/final}"
WAVLM_MODEL="${WAVLM_MODEL:-imprt/kushinada-hubert-large-laborotv2-asr}"

# OUTPUT_DIR="${OUTPUT_DIR:-${EXP_DIR}/ctc_decode}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_DIR}/jp_ctc_decode}"

# VOCAB_PATH="${VOCAB_PATH:-${EXP_DIR}/vocab.json}"
# BASE_MODEL="${BASE_MODEL:-microsoft/wavlm-large}"
SPLITS="${SPLITS:-test-clean test-other dev-clean dev-other}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATA_DEVICE="${DATA_DEVICE:-cpu}"
if [ "$DATA_DEVICE" == "cuda" ]; then
    NUM_WORKERS=0  
else
    NUM_WORKERS="${NUM_WORKERS:-4}"
fi

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ "$WAVLM_MODEL" == *.pt ]]; then
    if [ ! -f "$WAVLM_MODEL" ]; then
        echo "ERROR: WavLM checkpoint not found: $WAVLM_MODEL"
        exit 1
    fi
else
    if [ ! -d "$WAVLM_MODEL" ]; then
        echo "ERROR: WavLM model directory not found: $WAVLM_MODEL"
        exit 1
    fi
fi

if [ ! -d "$DATA_DIR" ] && [ ! -L "$DATA_DIR" ]; then
    echo "ERROR: LibriSpeech data directory not found: $DATA_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================"
echo "CTC Decoding with WavLM"
echo "========================================"
echo "Model:        $WAVLM_MODEL"
echo "Vocab:        ${VOCAB_PATH:-<from model dir>}"
echo "Data dir:     $DATA_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "Splits:       $SPLITS"
echo "Batch size:   $BATCH_SIZE"
echo "========================================"
echo ""

# Build extra args for .pt checkpoint
EXTRA_ARGS=""
if [[ "$WAVLM_MODEL" == *.pt ]]; then
    if [ -z "$VOCAB_PATH" ]; then
        echo "ERROR: VOCAB_PATH is required when using a .pt checkpoint"
        exit 1
    fi
    EXTRA_ARGS="--vocab-path $VOCAB_PATH --base-model $BASE_MODEL"
fi

CUDA_VISIBLE_DEVICES=$USE_DEVICE python3 pyscripts/decode_ctc.py \
    --wavlm-model "$WAVLM_MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --splits $SPLITS \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --data-device "$DATA_DEVICE" \
    $EXTRA_ARGS

echo ""
echo "========================================"
echo "CTC decoding completed!"
echo "Output files:"
for split in $SPLITS; do
    echo "  $OUTPUT_DIR/${split}_ctc.tsv"
done
echo ""
echo "Next steps:"
echo "  1. Create bias lists: bash make-bias-list.sh"
echo "  2. Run inference:     bash run-inference.sh"
echo "========================================"
