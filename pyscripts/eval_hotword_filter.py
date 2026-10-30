"""
Evaluate hotword filtering alone (no LLM, no GPU) on cached CTC decodes.

For every utterance the filter picks hotwords from its ``bias_list`` given the
CTC hypothesis, and the picks are compared with the ``true_hotwords`` (rare
words of the reference) by exact string match. Counts are summed over all
utterances (micro average):

* recall    = correct picks / true hotwords  – did the right words reach the prompt?
* precision = correct picks / picked words   – utterances without any true
  hotword still count their picks, so read precision as a relative number.

Filter settings come from the ``hotword_filter`` section of ``--infer-config``
(same keys as ``src/inference/inference.py``), else the filter's own defaults;
the CLI flags override both.

Usage:
    python pyscripts/eval_hotword_filter.py --filter-method ja \
        --infer-config configs/inference_ja.yaml \
        --common-words-path data/lexicon/csj/common_words_5k.txt \
        --ctc-file exp/whisper-ft-ctc_ja/ctc_decode/eval{1,2,3}_whisper_like_clean_ctc.tsv \
        --bias-list-json data/lexicon/csj/hotwords/eval{1,2,3}_whisper_like_clean_bias_list.json

    # Try another precision floor without editing the config; dump the misses
    ... --probability-threshold 0.6 --misses-out misses.jsonl
"""

import argparse
import importlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.inference_utils import load_bias_list, load_ctc_file  # noqa: E402

logger = logging.getLogger(__name__)

# hotword_filter keys passed to every filter / only to the Japanese one
COMMON_KEYS = ("common_words_path", "n_gram", "word_num", "probability_threshold")
JA_KEYS = (
    "tokenizer_backend",
    "reading_weight",
    "surface_bonus",
    "include_homophones",
    "max_homophones",
    "per_word_scoring",
)


@dataclass
class FilterStats:
    utts: int = 0
    true: int = 0
    picked: int = 0
    correct: int = 0

    def __add__(self, other: "FilterStats") -> "FilterStats":
        return FilterStats(
            self.utts + other.utts,
            self.true + other.true,
            self.picked + other.picked,
            self.correct + other.correct,
        )

    @property
    def recall(self) -> Optional[float]:
        return self.correct / self.true if self.true else None

    @property
    def precision(self) -> Optional[float]:
        return self.correct / self.picked if self.picked else None

    def to_dict(self) -> Dict:
        return {**asdict(self), "recall": self.recall, "precision": self.precision}

    def __str__(self) -> str:
        def pct(x: Optional[float]) -> str:
            return "n/a" if x is None else f"{x * 100:.2f}%"

        avg = self.picked / self.utts if self.utts else 0.0
        return (
            f"recall: {pct(self.recall)} ({self.correct}/{self.true} true hotwords)\n"
            f"precision: {pct(self.precision)} ({self.correct}/{self.picked} picked)\n"
            f"picked per utt: {avg:.2f}"
        )


def evaluate_split(
    hotword_filter, hyps: Dict[str, str], bias_data: Dict[str, Dict]
) -> Tuple[FilterStats, List[Dict]]:
    """Run the filter on every utterance of one split; return the stats and
    the utterances that missed at least one true hotword."""
    stats = FilterStats()
    misses: List[Dict] = []
    no_hyp = 0
    for uid, entry in bias_data.items():
        hyp = hyps.get(uid, "")
        no_hyp += not hyp
        picked = set(hotword_filter.filter_hotwords(hyp, entry["bias_list"])) if hyp else set()
        true = set(entry["true_hotwords"])
        stats += FilterStats(1, len(true), len(picked), len(picked & true))
        if true - picked:
            misses.append(
                {"utt_id": uid, "hyp": hyp, "missed": sorted(true - picked), "picked": sorted(picked)}
            )
    if no_hyp:
        logger.warning(f"{no_hyp} utterance(s) have no CTC hypothesis; nothing picked for them")
    return stats, misses


def build_filter(args):
    cfg: Dict = {}
    if args.infer_config:
        with open(args.infer_config, encoding="utf-8") as f:
            cfg = (yaml.safe_load(f) or {}).get("hotword_filter") or {}
    keys = COMMON_KEYS + (JA_KEYS if args.filter_method == "ja" else ())
    kwargs = {k: cfg[k] for k in keys if k in cfg}
    for key in COMMON_KEYS:
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value

    module = importlib.import_module(f"src.inference.hotword_filter_{args.filter_method}")
    # The filters log one INFO line per utterance
    logging.getLogger(module.__name__).setLevel(logging.WARNING)
    logger.info(f"HotwordFilter [{args.filter_method}]: {kwargs}")
    return module.HotwordFilter(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ctc-file", nargs="+", required=True, help="<split>_ctc.tsv, one per split")
    parser.add_argument(
        "--bias-list-json", nargs="+", required=True, help="<split>_bias_list.json, same order as --ctc-file"
    )
    parser.add_argument("--filter-method", choices=["F1", "F2", "F3", "ja"], default="ja")
    parser.add_argument("--infer-config", help="Inference YAML with a hotword_filter section")
    parser.add_argument("--common-words-path", default=None)
    parser.add_argument("--n-gram", type=int, default=None)
    parser.add_argument("--word-num", type=int, default=None)
    parser.add_argument("--probability-threshold", type=float, default=None)
    parser.add_argument("--output", help="Write all results to this JSON file")
    parser.add_argument(
        "--misses-out", help="Write utterances that missed a true hotword (JSON lines) for inspection"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(args.ctc_file) != len(args.bias_list_json):
        parser.error("--ctc-file and --bias-list-json need one file per split, in the same order")

    hotword_filter = build_filter(args)
    misses_f = open(args.misses_out, "w", encoding="utf-8") if args.misses_out else None
    results: Dict[str, FilterStats] = {}
    for ctc_path, bias_path in zip(args.ctc_file, args.bias_list_json):
        name = Path(bias_path).name
        split = name[: -len("_bias_list.json")] if name.endswith("_bias_list.json") else Path(bias_path).stem
        stats, misses = evaluate_split(hotword_filter, load_ctc_file(ctc_path), load_bias_list(bias_path))
        results[split] = stats
        print(f"== {split} ({stats.utts} utts, filter={args.filter_method})\n{stats}")
        if misses_f is not None:
            for m in misses:
                misses_f.write(json.dumps({"split": split, **m}, ensure_ascii=False) + "\n")

    if len(results) > 1:
        results["overall"] = sum(results.values(), FilterStats())
        print(f"== overall\n{results['overall']}")
    if misses_f is not None:
        misses_f.close()
        logger.info(f"Misses written to {args.misses_out}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
        logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
