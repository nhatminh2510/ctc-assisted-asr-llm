"""Compute and display LibriSpeech dataset statistics.

Reports per-split and overall:
  - Total utterances
  - Total words
  - Total audio duration (hours)
  - Duration distribution (min, max, mean, median, std, percentiles)

Usage:
    python pyscripts/dataset_stats.py [--data_dir DATA_DIR] [--splits SPLIT1 SPLIT2 ...]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Defaults matching the project config
DEFAULT_DATA_DIR = "data/LibriSpeech"
ALL_SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]


def gather_samples(split_path: Path):
    """Yield (audio_path, transcript) for every utterance in a split."""
    for speaker_dir in sorted(split_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            trans_file = (
                chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            )
            if not trans_file.exists():
                continue
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utt_id, transcript = parts
                        audio_file = chapter_dir / f"{utt_id}.flac"
                        if audio_file.exists():
                            yield audio_file, transcript


def compute_split_stats(data_dir: Path, split: str):
    """Return stats dict for one split."""
    split_path = data_dir / split
    if not split_path.exists():
        return None

    durations = []
    total_words = 0
    total_utts = 0

    samples = list(gather_samples(split_path))
    for audio_path, transcript in tqdm(samples, desc=f"  {split}", leave=False):
        info = sf.info(str(audio_path))
        durations.append(info.duration)
        total_words += len(transcript.split())
        total_utts += 1

    if total_utts == 0:
        return None

    dur = np.array(durations)
    return {
        "split": split,
        "utterances": total_utts,
        "total_words": total_words,
        "total_hours": dur.sum() / 3600,
        "dur_min": dur.min(),
        "dur_max": dur.max(),
        "dur_mean": dur.mean(),
        "dur_median": np.median(dur),
        "dur_std": dur.std(),
        "dur_p5": np.percentile(dur, 5),
        "dur_p25": np.percentile(dur, 25),
        "dur_p75": np.percentile(dur, 75),
        "dur_p95": np.percentile(dur, 95),
    }


def print_split_stats(s):
    """Pretty-print one split's stats."""
    print(f"\n{'=' * 60}")
    print(f"  Split: {s['split']}")
    print(f"{'=' * 60}")
    print(f"  Utterances      : {s['utterances']:>10,}")
    print(f"  Total words     : {s['total_words']:>10,}")
    print(f"  Total duration  : {s['total_hours']:>10.2f} h")
    print(f"  Duration (sec)  :")
    print(f"    min / max     : {s['dur_min']:>8.2f} / {s['dur_max']:.2f}")
    print(f"    mean ± std    : {s['dur_mean']:>8.2f} ± {s['dur_std']:.2f}")
    print(f"    median        : {s['dur_median']:>8.2f}")
    print(f"    p5 / p25      : {s['dur_p5']:>8.2f} / {s['dur_p25']:.2f}")
    print(f"    p75 / p95     : {s['dur_p75']:>8.2f} / {s['dur_p95']:.2f}")


def print_overall(all_stats):
    """Print aggregated stats across all splits."""
    total_utts = sum(s["utterances"] for s in all_stats)
    total_words = sum(s["total_words"] for s in all_stats)
    total_hours = sum(s["total_hours"] for s in all_stats)

    print(f"\n{'#' * 60}")
    print(f"  OVERALL ({len(all_stats)} splits)")
    print(f"{'#' * 60}")
    print(f"  Total utterances : {total_utts:>10,}")
    print(f"  Total words      : {total_words:>10,}")
    print(f"  Total duration   : {total_hours:>10.2f} h")
    print(f"{'#' * 60}")


def main():
    parser = argparse.ArgumentParser(description="LibriSpeech dataset statistics")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to analyse (default: all found)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    splits = args.splits or [s for s in ALL_SPLITS if (data_dir / s).exists()]
    if not splits:
        print("No splits found.", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {len(splits)} split(s) in {data_dir} …")

    all_stats = []
    for split in splits:
        stats = compute_split_stats(data_dir, split)
        if stats is None:
            print(f"  [skip] {split} — not found or empty")
            continue
        print_split_stats(stats)
        all_stats.append(stats)

    if len(all_stats) > 1:
        print_overall(all_stats)


if __name__ == "__main__":
    main()
