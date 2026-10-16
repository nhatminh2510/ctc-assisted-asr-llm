import json
import logging
from pathlib import Path
from typing import List, Dict, Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bias list / CTC file loaders
# ---------------------------------------------------------------------------
def load_bias_list(bias_list_json: str) -> Dict[str, Dict]:
    """
    Load per-utterance bias list JSON.

    Returns a dict: utt_id -> {
        "bias_list": [...],
        "true_hotwords": [...],
        "transcription": "...",
    }
    """
    with open(bias_list_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for entry in data:
        uid = entry["utt_id"]
        result[uid] = {
            "bias_list": entry.get("bias_list", []),
            "true_hotwords": entry.get("true_hotwords", []),
            "transcription": entry.get("transcription", ""),
        }
    logger.info(f"Loaded bias list for {len(result)} utterances from {bias_list_json}")
    return result


def load_bias_lists(paths: List[str]) -> Dict[str, Dict]:
    """Load and merge multiple bias list JSONs (one per split)."""
    merged: Dict[str, Dict] = {}
    for p in paths:
        merged.update(load_bias_list(p))
    logger.info(
        f"Merged bias lists: {len(merged)} utterances from {len(paths)} file(s)"
    )
    return merged


def biasing_words_for(bias_data: Dict[str, Dict], utt_ids: List[str]) -> List[List[str]]:
    """Per-utterance biasing words for B-WER: the ``true_hotwords`` (rare words
    of the reference), as in is21_deep_bias. Utterances missing from
    ``bias_data`` get none (all their words count as unbiased)."""
    missing = [u for u in utt_ids if u not in bias_data]
    if missing:
        logger.warning(
            f"{len(missing)}/{len(utt_ids)} utterances have no bias list entry "
            f"(e.g. {missing[0]}); their words all count as unbiased in B-WER."
        )
    return [bias_data.get(u, {}).get("true_hotwords", []) for u in utt_ids]


def load_ctc_file(ctc_path: str) -> Dict[str, str]:
    """
    Load CTC pseudo-label file (TSV: utt_id\\tctc_text).

    Returns dict: utt_id -> ctc_decoded_text
    """
    results: Dict[str, str] = {}
    with open(ctc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                results[parts[0]] = parts[1]
    logger.info(f"Loaded {len(results)} CTC decodes from {ctc_path}")
    return results


def load_ctc_files(paths: List[str]) -> Dict[str, str]:
    """Load and merge multiple CTC decode files (one per split)."""
    merged: Dict[str, str] = {}
    for p in paths:
        merged.update(load_ctc_file(p))
    logger.info(f"Merged CTC files: {len(merged)} utterances from {len(paths)} file(s)")
    return merged


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------
def save_ctc_file(results: Dict[str, str], output_path: str) -> None:
    """Save CTC decode results as TSV: utt_id\\tctc_text"""
    with open(output_path, "w", encoding="utf-8") as f:
        for uid, text in sorted(results.items()):
            f.write(f"{uid}\t{text}\n")
    logger.info(f"Saved {len(results)} CTC decodes to {output_path}")


def save_results(results: Dict[str, Any], output_dir: str, split: str) -> None:
    """Persist evaluation results to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Summary JSON
    summary = {
        "split": split,
        "metric": results.get("metric", "wer"),
        "wer": results["wer"],
        "num_samples": len(results["hyps"]),
    }
    if "bias_wer" in results:
        summary["bias_wer"] = results["bias_wer"]
    summary_path = out / f"{split}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # 2. Per-utterance details
    details = []
    for uid, hyp, ref, raw_hyp, raw_ref in zip(
        results["utt_ids"],
        results["hyps"],
        results["refs"],
        results["raw_hyps"],
        results["raw_refs"],
    ):
        details.append(
            {
                "utt_id": uid,
                "hyp": hyp,
                "ref": ref,
                "raw_hyp": raw_hyp,
                "raw_ref": raw_ref,
            }
        )
    details_path = out / f"{split}_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    logger.info(f"Details saved to {details_path}")

    # 3. Plain text hyp/ref files (for external scoring tools)
    with open(out / f"{split}_hyp.txt", "w", encoding="utf-8") as fh, open(
        out / f"{split}_ref.txt", "w", encoding="utf-8"
    ) as fr:
        for uid, hyp, ref in zip(results["utt_ids"], results["hyps"], results["refs"]):
            fh.write(f"{uid} {hyp}\n")
            fr.write(f"{uid} {ref}\n")
