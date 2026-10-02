#!/bin/bash
# =============================================================================
# Create bias (hotword) lists for LibriSpeech evaluation
#
# Usage:
#   bash make-bias-list.sh
#
#   # Custom settings
#   NUM_DISTRACTORS=200 bash make-bias-list.sh
#
#   # Japanese (CSJ): word frequencies computed from the train split
#   DATA_DIR=data/csj LANGUAGE=ja LEXICON_DIR=data/lexicon/csj \
#       OUTPUT_DIR=data/lexicon/csj/hotwords \
#       WORD_FREQ_PATH=data/lexicon/csj/all_words.count.txt \
#       COMMON_WORDS_PATH=data/lexicon/csj/common_words_5k.txt \
#       ALL_RARE_WORDS_PATH=data/lexicon/csj/all_rare_words.txt \
#       FREQ_SPLITS=train_nodup_whisper_like_clean \
#       TEST_SPLITS="eval1_whisper_like_clean eval2_whisper_like_clean eval3_whisper_like_clean" \
#       bash make-bias-list.sh
# =============================================================================

set -e

DATA_DIR="${DATA_DIR:-/home/25/minh/ctc-assisted-asr-llm/data/LibriSpeech}"
DATASET="${DATASET:-auto}"        # auto | librispeech | csj
LANGUAGE="${LANGUAGE:-en}"        # en | ja (ja needs fugashi/SudachiPy/pykakasi)
# Optional: (re)compute WORD_FREQ_PATH from these training splits first
FREQ_SPLITS="${FREQ_SPLITS:-}"
LEXICON_DIR="${LEXICON_DIR:-/home/25/minh/ctc-assisted-asr-llm/data/lexicon}"
mkdir -p "$LEXICON_DIR"
OUTPUT_DIR="${OUTPUT_DIR:-${LEXICON_DIR}/LibriSpeech/hotwords}"

# Test splits to create bias lists for 
TEST_SPLITS="${TEST_SPLITS:-test-clean test-other}"

# Pre-computed word frequency file (TSV: word\tcount)
WORD_FREQ_PATH="${WORD_FREQ_PATH:-${LEXICON_DIR}/all_words.count.org.txt}"

# Number of distractor (non-target) hotwords per utterance
NUM_DISTRACTORS="${NUM_DISTRACTORS:-100}"

SEED="${SEED:-42}"

# Common words file (top-k most frequent training words)
COMMON_WORDS_PATH="${COMMON_WORDS_PATH:-${LEXICON_DIR}/LibriSpeech/common_words_5k.txt}"
COMMON_WORDS_TOP_K="${COMMON_WORDS_TOP_K:-5000}"

# Output path for all rare words
ALL_RARE_WORDS_PATH="${ALL_RARE_WORDS_PATH:-${LEXICON_DIR}/LibriSpeech/all_rare_words.txt}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [ ! -d "$DATA_DIR" ] && [ ! -L "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
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

FREQ_ARGS=""
if [ -n "$FREQ_SPLITS" ]; then
    FREQ_ARGS="--freq-splits $FREQ_SPLITS"
fi

python3 pyscripts/create_bias_list.py \
    --data-dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --language "$LANGUAGE" \
    $FREQ_ARGS \
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
