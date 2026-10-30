from typing import Callable, List, Optional, Tuple
import logging
import unicodedata

import jiwer


logger = logging.getLogger(__name__)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Tính Word Error Rate (WER).

    Args:
        predictions: Danh sách các phiên âm dự đoán
        references: Danh sách các phiên âm ground truth

    Returns:
        wer: Word Error Rate
    """
    try:

        wer = jiwer.wer(references, predictions)
        return wer
    except ImportError:
        raise ImportError(
            "jiwer is required for WER computation. Install with: pip install jiwer"
        )
    except Exception as e:
        logger.error(f"Error computing WER: {e}")
        raise


def normalize_japanese(text: str) -> str:
    """Normalise Japanese text for CER scoring.

    NFKC (full-width Latin/digits -> half-width, half-width kana -> full-width),
    lower-case, then drop whitespace, punctuation, symbols and control chars.
    The long-vowel mark ``ー`` (category Lm) is kept.
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] not in ("P", "S", "Z", "C")
    )


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Corpus-level Character Error Rate: total edits / total reference chars.

    Implemented with Levenshtein distances directly (instead of ``jiwer.cer``)
    so an empty reference after normalisation does not raise.
    """
    import Levenshtein

    edits = sum(Levenshtein.distance(h, r) for h, r in zip(predictions, references))
    ref_chars = sum(len(r) for r in references)
    return edits / max(ref_chars, 1)


def score_transcripts(
    hyps: List[str],
    refs: List[str],
    language: str = "en",
    normalizer: Optional[Callable[[str], str]] = None,
) -> Tuple[str, float, List[str], List[str]]:
    """Normalise and score hypotheses against references, per language.

    * ``en``: ``normalizer`` (e.g. Whisper's ``BasicTextNormalizer``) + WER.
    * ``ja``: :func:`normalize_japanese` + CER. Japanese has no spaces, so WER
      would only count errors between punctuation-delimited chunks.

    Returns ``(metric_name, error_rate, normalised_hyps, normalised_refs)``.
    Raises ``ValueError`` when there is nothing to score, so an empty/misloaded
    dataset never masquerades as a perfect 0% error rate.
    """
    if not hyps or not refs:
        raise ValueError(
            "No hypotheses/references to score — the dataset is empty "
            "(check --data-dir / --splits / --dataset)."
        )
    if language == "ja":
        h = [normalize_japanese(x) for x in hyps]
        r = [normalize_japanese(x) for x in refs]
        return "cer", compute_cer(h, r), h, r

    norm = normalizer or (lambda s: s)
    h = [norm(x.strip()) for x in hyps]
    r = [norm(x.strip()) if isinstance(x, str) else [norm(y) for y in x] for x in refs]
    return "wer", compute_wer(h, r), h, r
