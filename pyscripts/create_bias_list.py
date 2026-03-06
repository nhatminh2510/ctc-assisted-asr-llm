"""
Create bias (hotword) lists for LibriSpeech evaluation.

A bias list consists of:
  - True hotwords: rare words actually present in each utterance's transcription
  - Distractors:   other rare words NOT in the utterance (to simulate real-world noise)

Words are considered "rare" if they are NOT among the top-k most frequent words
(i.e. they appear in all_words.count.txt but not in common_words_5k.txt).

Outputs:
  1. Per-utterance JSON  – contains true_hotwords, distractors, and shuffled bias_list
  2. Global rare word list (plain text, one word per line)

Usage:
    python scripts/create_bias_list.py \
        --data-dir data/LibriSpeech \
        --output-dir data/hotwords \
        --test-splits test-clean test-other \
        --word-freq-path data/all_words.count.txt \
        --num-distractors 100

Based on: https://arxiv.org/pdf/2411.06437
Reference: https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/contextual_asr
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# LibriSpeech transcript loading
# ---------------------------------------------------------------------------
def load_librispeech_transcripts(data_dir: str, split: str) -> Dict[str, str]:
    """
    Load LibriSpeech transcriptions from .trans.txt files.

    Args:
        data_dir: Root LibriSpeech directory
        split: Split name (e.g. 'train-clean-100', 'test-clean')

    Returns:
        dict mapping utt_id -> transcription (UPPER CASE)
    """
    split_path = Path(data_dir) / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    transcripts: Dict[str, str] = {}
    for trans_file in sorted(split_path.rglob("*.trans.txt")):
        with open(trans_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text

    return transcripts


# ---------------------------------------------------------------------------
# Word frequency loading
# ---------------------------------------------------------------------------
def load_word_frequencies(freq_path: str) -> Counter:
    """
    Load pre-computed word frequencies from a TSV file.

    Expected format (one word per line):
        word\tcount

    Words are stored as UPPER CASE to match LibriSpeech transcriptions.

    Args:
        freq_path: Path to the word frequency file (e.g. data/all_words.count.txt)

    Returns:
        Counter mapping word (UPPER CASE) -> count
    """
    freq_file = Path(freq_path)
    if not freq_file.exists():
        raise FileNotFoundError(f"Word frequency file not found: {freq_path}")

    word_counter: Counter = Counter()
    with open(freq_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                word, count = parts
                word_counter[word] = int(count)
    return word_counter

def compute_word_frequencies(data_dir: str, output_path: str) -> str:
    """
    Compute word frequencies from all LibriSpeech transcriptions and save to a TSV file.

    Output format (one word per line):
        word\tcount

    Words are stored as UPPER CASE to match LibriSpeech transcriptions.

    Args:
        data_dir: Root LibriSpeech directory
        output_path: Path to save the word frequency file (e.g. data/all_words.count.txt)
    """
    word_counter: Counter = Counter()

    # Process all splits (train + test) to get comprehensive frequencies
    for split in ["train-clean-100", "train-clean-360", "train-other-500"]:
        print(f"Processing split for word frequencies: {split}")
        transcripts = load_librispeech_transcripts(data_dir, split)
        for text in transcripts.values():
            words = text.split()
            word_counter.update(w.lower() for w in words)

    # Save word frequencies to output path
    with open(output_path, "w", encoding="utf-8") as f:
        for word, count in sorted(word_counter.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{word}\t{count}\n")
    print(f"Saved word frequencies for {len(word_counter)} words to {output_path}")
    return output_path

# ---------------------------------------------------------------------------
# Rare word extraction
# ---------------------------------------------------------------------------
def build_rare_word_set(
    word_freq: Counter,
    common_words: Set[str],
) -> Set[str]:
    """
    Build the global set of rare words as:
        rare = {all words in word_freq} - {common words}

    Both sides are compared in UPPER CASE.

    Args:
        word_freq: Counter mapping word (UPPER CASE) -> count
        common_words: Set of common words (UPPER CASE)

    Returns:
        Set of rare words (UPPER CASE)
    """
    return set(word_freq.keys()) - common_words


def find_rare_words(
    transcripts: Dict[str, str],
    rare_word_set: Set[str],
) -> Set[str]:
    """
    Find rare words that actually appear in test transcriptions.

    Args:
        transcripts: dict mapping utt_id -> transcription (UPPER CASE)
        rare_word_set: Global set of rare words (UPPER CASE)

    Returns:
        Subset of rare_word_set that appears in at least one transcription
    """
    all_test_words: Set[str] = set()
    for text in transcripts.values():
        all_test_words.update(text.split())

    return all_test_words & rare_word_set


# ---------------------------------------------------------------------------
# Per-utterance bias list construction
# ---------------------------------------------------------------------------
def create_per_utterance_bias_lists(
    transcripts: Dict[str, str],
    rare_words: Set[str],
    num_distractors: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    For each utterance build a bias list:
      - true_hotwords: rare words that appear in the utterance
      - distractors:   randomly sampled rare words that do NOT appear

    Returns list of dicts with keys:
        utt_id, transcription, true_hotwords, distractors, bias_list
    """
    rng = random.Random(seed)
    rare_sorted = sorted(rare_words)

    results = []
    for utt_id, text in sorted(transcripts.items()):
        words_in_utt = set(text.split())

        true_hotwords = sorted(words_in_utt & rare_words)

        distractor_pool = [w for w in rare_sorted if w not in words_in_utt]
        n = min(num_distractors, len(distractor_pool))
        distractors = sorted(rng.sample(distractor_pool, n))

        bias_list = true_hotwords + distractors
        rng.shuffle(bias_list)

        results.append(
            {
                "utt_id": utt_id,
                "transcription": text,
                "true_hotwords": true_hotwords,
                "num_true_hotwords": len(true_hotwords),
                "distractors": distractors,
                "bias_list": bias_list,
            }
        )

    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_common_words(
    word_freq: Counter,
    output_path: str,
    top_k: int = 5000,
) -> str:
    """
    Save the top-k most frequent training words to a text file (one word per line, lowercase).

    This file is used by HotwordFilter to exclude common words from bias lists
    during inference.

    Args:
        word_freq: Counter mapping word (UPPER CASE) -> count
        output_path: Path to the output file
        top_k: Number of most frequent words to save (default: 5000)

    Returns:
        Path to the saved file
    """
    most_common = word_freq.most_common(top_k)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for word, _ in most_common:
            f.write(word.lower() + "\n")

    return output_path


def save_all_rare_words(
    rare_word_set: Set[str],
    output_path: str,
) -> str:
    """
    Save all rare words (all_words - common_words) to a text file, one per line (lowercase).

    Args:
        rare_word_set: Set of rare words (UPPER CASE)
        output_path: Path to the output file

    Returns:
        Path to the saved file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted(rare_word_set):
            f.write(word.lower() + "\n")
    return output_path


def save_bias_lists(
    results: List[Dict],
    rare_words: Set[str],
    output_dir: str,
    split: str,
) -> Tuple[str, str]:
    """Save per-utterance JSON and global rare word list."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Per-utterance bias lists
    json_path = out / f"{split}_bias_list.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 2. Global flat rare words (one per line)
    global_path = out / f"{split}_rare_words.txt"
    with open(global_path, "w") as f:
        for w in sorted(rare_words):
            f.write(w + "\n")

    return str(json_path), str(global_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create bias/hotword lists for LibriSpeech evaluation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/LibriSpeech",
        help="Root directory of LibriSpeech data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hotwords",
        help="Directory to save bias lists",
    )
    parser.add_argument(
        "--test-splits",
        type=str,
        nargs="+",
        default=["test-clean", "test-other"],
        help="Test splits to create bias lists for",
    )
    parser.add_argument(
        "--word-freq-path",
        type=str,
        default="data/all_words.count.txt",
        help="Path to pre-computed word frequency file (TSV: word\\tcount)",
    )
    parser.add_argument(
        "--all-rare-words-path",
        type=str,
        default="data/all_rare_words.txt",
        help="Path to save all rare words file (default: data/all_rare_words.txt)",
    )
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=100,
        help="Number of distractor hotwords per utterance (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for distractor sampling",
    )
    parser.add_argument(
        "--common-words-path",
        type=str,
        default="data/common_words_5k.txt",
        help="Path to save common words file (default: data/common_words_5k.txt)",
    )
    parser.add_argument(
        "--common-words-top-k",
        type=int,
        default=5000,
        help="Number of most frequent words to save (default: 5000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: Load word frequencies
    print(f"Loading word frequencies from: {args.word_freq_path}")
    word_freq = load_word_frequencies(args.word_freq_path)
    print(f"  Vocabulary size: {len(word_freq):,}")
    print(f"  Total tokens:    {sum(word_freq.values()):,}")

    # Step 1b: Save common words file
    common_path = save_common_words(
        word_freq, args.common_words_path, top_k=args.common_words_top_k
    )
    print(f"  Saved top {args.common_words_top_k} common words to {common_path}")

    # Step 1c: Build and save global rare word set
    common_words_upper: Set[str] = set()
    with open(common_path, "r", encoding="utf-8") as f:
        common_words_upper = {line.strip() for line in f if line.strip()}
    rare_word_set = build_rare_word_set(word_freq, common_words_upper)
    rare_path = save_all_rare_words(rare_word_set, args.all_rare_words_path)
    print(f"  Rare words (all_words - common_words): {len(rare_word_set):,}")
    print(f"  Saved to {rare_path}")

    # Step 2: Process each test split
    for split in args.test_splits:
        print(f"\nProcessing split: {split}")

        transcripts = load_librispeech_transcripts(args.data_dir, split)
        print(f"  Utterances: {len(transcripts):,}")

        rare_words = find_rare_words(transcripts, rare_word_set)
        print(f"  Rare words in {split}: {len(rare_words):,}")

        results = create_per_utterance_bias_lists(
            transcripts,
            rare_words,
            num_distractors=args.num_distractors,
            seed=args.seed,
        )

        utts_with_hw = sum(1 for r in results if r["num_true_hotwords"] > 0)
        total_hw = sum(r["num_true_hotwords"] for r in results)
        avg_hw = total_hw / max(utts_with_hw, 1)

        print(f"  Utterances with >= 1 rare word: {utts_with_hw}/{len(results)}")
        print(f"  Total true hotword occurrences: {total_hw}")
        print(f"  Avg hotwords per biased utt:    {avg_hw:.1f}")

        json_path, global_path = save_bias_lists(
            results, rare_words, args.output_dir, split
        )
        print(f"  Per-utterance bias list: {json_path}")
        print(f"  Global rare word list:   {global_path}")

    print(f"\nDone! Bias lists saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
    # For testing word frequency computation:
    # compute_word_frequencies(
    #     data_dir="data/LibriSpeech",
    #     output_path="data/all_words.count.txt",
    # )
