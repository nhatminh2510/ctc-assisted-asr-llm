"""
Compute WER, U-WER and B-WER (R-WER) for saved inference results.

Same metrics as facebookresearch/fbai-speech ``is21_deep_bias/score.py``
(see ``src/utils/bias_wer.py``): B-WER is the error rate on the rare words of
each reference (``true_hotwords`` of the bias list), U-WER on all other words.
Japanese is split into words with a morphological analyzer (fugashi by default).

Inputs are the ``<split>_details.json`` files written by
``src/inference/inference.py`` (or their directory), plus the bias list JSONs.

Usage:
    python pyscripts/score_bias_wer.py --language ja \
        --details exp/whisper-ft-ctc_llm_contextual/inference/filter-ja \
        --bias-list-json data/lexicon/csj/hotwords/eval{1,2,3}_whisper_like_clean_bias_list.json

    # English (LibriSpeech), Whisper BasicTextNormalizer as in inference.py
    python pyscripts/score_bias_wer.py --language en \
        --details exp/<exp>/inference/gt/test-clean_details.json \
        --bias-list-json data/lexicon/LibriSpeech/bias_list_100/test-clean_bias_list.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.bias_wer import BiasWERResult, BiasWERScorer, align  # noqa: E402
from src.utils.inference_utils import biasing_words_for, load_bias_lists  # noqa: E402

logger = logging.getLogger(__name__)


def resolve_details(paths: List[str]) -> List[Path]:
    """Expand directories into their ``*_details.json`` files."""
    files: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found = sorted(p.glob("*_details.json"))
            if not found:
                raise FileNotFoundError(f"No *_details.json in {p}")
            files.extend(found)
        else:
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--details",
        nargs="+",
        required=True,
        help="<split>_details.json file(s), or inference output dir(s)",
    )
    parser.add_argument("--bias-list-json", nargs="+", required=True)
    parser.add_argument("--language", choices=["en", "ja"], default="ja")
    parser.add_argument(
        "--word-backend",
        default="auto",
        choices=["auto", "fugashi", "sudachipy", "pykakasi", "char"],
        help="Japanese word segmenter (default: auto = fugashi > sudachipy > ...). "
        "Use the one the bias list was built with.",
    )
    parser.add_argument("--output", help="Write all results to this JSON file")
    parser.add_argument(
        "--alignment-out",
        help="Write per-utterance words and alignment (JSON lines) for inspection",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    normalizer = None
    if args.language == "en":
        from whisper.normalizers import BasicTextNormalizer

        normalizer = BasicTextNormalizer()
    scorer = BiasWERScorer(args.language, normalizer, args.word_backend)
    bias_data = load_bias_lists(args.bias_list_json)

    align_f = open(args.alignment_out, "w", encoding="utf-8") if args.alignment_out else None
    results = {}
    for path in resolve_details(args.details):
        split = path.name[: -len("_details.json")] if path.name.endswith("_details.json") else path.stem
        details = json.load(open(path, encoding="utf-8"))
        utt_ids = [d["utt_id"] for d in details]
        hyps = [d.get("raw_hyp", d["hyp"]) for d in details]
        refs = [d.get("raw_ref", d["ref"]) for d in details]
        words = biasing_words_for(bias_data, utt_ids)
        results[split] = scorer.score(hyps, refs, words)
        print(f"== {split} ({len(details)} utts, {scorer.split.name} words)\n{results[split]}")

        if align_f is not None:
            for uid, hyp, ref, w in zip(utt_ids, hyps, refs, words):
                hyp_t, ref_t, bias = scorer.tokenize(hyp, ref, w)
                pairs = [
                    [code.name, None if ri is None else ref_t[ri], None if hi is None else hyp_t[hi]]
                    for code, ri, hi in align(ref_t, hyp_t)
                ]
                align_f.write(
                    json.dumps(
                        {"utt_id": uid, "split": split, "biasing_words": sorted(bias), "alignment": pairs},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if len(results) > 1:
        results["overall"] = sum(results.values(), BiasWERResult())
        print(f"== overall\n{results['overall']}")
    if align_f is not None:
        align_f.close()
        logger.info(f"Alignments written to {args.alignment_out}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
        logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
