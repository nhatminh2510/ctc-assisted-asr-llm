#!/bin/bash
# =============================================================================
# Create bias (hotword) lists for LibriSpeech evaluation
#
# Usage:
#   bash make-bias-list.sh
#
#   # Custom settings
#   NUM_DISTRACTORS=200 bash make-bias-list.sh
# =============================================================================

set -e

DATA_DIR="${DATA_DIR:-/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-/home/25/minh/CTC-assisted-ASR-LLM/data/processed}"
mkdir -p "$PROCESSED_DATA_DIR"
OUTPUT_DIR="${OUTPUT_DIR:-${PROCESSED_DATA_DIR}/LibriSpeech/hotwords}"

# Test splits to create bias lists for 
TEST_SPLITS="${TEST_SPLITS:-test-clean test-other}"

# Pre-computed word frequency file (TSV: word\tcount)
WORD_FREQ_PATH="${WORD_FREQ_PATH:-${PROCESSED_DATA_DIR}/all_words.count.org.txt}"

# Number of distractor (non-target) hotwords per utterance
NUM_DISTRACTORS="${NUM_DISTRACTORS:-100}"

SEED="${SEED:-42}"

# Common words file (top-k most frequent training words)
COMMON_WORDS_PATH="${COMMON_WORDS_PATH:-${PROCESSED_DATA_DIR}/LibriSpeech/common_words_5k.txt}"
COMMON_WORDS_TOP_K="${COMMON_WORDS_TOP_K:-5000}"

# Output path for all rare words
ALL_RARE_WORDS_PATH="${ALL_RARE_WORDS_PATH:-${PROCESSED_DATA_DIR}/LibriSpeech/all_rare_words.txt}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [ ! -d "$DATA_DIR" ] && [ ! -L "$DATA_DIR" ]; then
    echo "ERROR: LibriSpeech data directory not found: $DATA_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================"
echo "Creating bias lists"
echo "========================================"
echo "Data dir:         $DATA_DIR"
echo "Output dir:       $OUTPUT_DIR"
echo "Test splits:      $TEST_SPLITS"
echo "Word freq file:   $WORD_FREQ_PATH"
echo "Num distractors:  $NUM_DISTRACTORS"
echo "Common words:     $COMMON_WORDS_PATH (top $COMMON_WORDS_TOP_K)"
echo "Rare words:       $ALL_RARE_WORDS_PATH"
echo "========================================"
echo ""

python3 scripts/create_bias_list.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --test-splits $TEST_SPLITS \
    --word-freq-path "$WORD_FREQ_PATH" \
    --all-rare-words-path "$ALL_RARE_WORDS_PATH" \
    --num-distractors "$NUM_DISTRACTORS" \
    --seed "$SEED" \
    --common-words-path "$COMMON_WORDS_PATH" \
    --common-words-top-k "$COMMON_WORDS_TOP_K"

echo ""
echo "========================================"
echo "Done! Bias lists saved to $OUTPUT_DIR/"
echo ""
echo "To use with inference:"
echo "  bash run-inference.sh  # with BIAS_LIST_JSON set"
echo "  e.g.:"
echo "  BIAS_LIST_JSON=$OUTPUT_DIR/test-clean_bias_list.json bash run-inference.sh"
echo "========================================"
