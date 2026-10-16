"""WER / U-WER / B-WER (biased-word WER) for contextual ASR.

Port of ``score.py`` from facebookresearch/fbai-speech ``is21_deep_bias``
(Le et al., "Contextualized Streaming End-to-End Speech Recognition with
Trie-Based Deep Biasing and Shallow Fusion", Interspeech 2021):
https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias

Every utterance is aligned once (costs: substitution 4, insertion/deletion 3,
same tie-breaking as the reference) and each edit is attributed to one of:

* **WER**   – all reference words.
* **U-WER** – reference words *not* in the utterance's biasing words, plus
  insertions of non-biasing words.
* **B-WER** – reference words in the utterance's biasing words, plus
  insertions of biasing words. The biasing words are the rare words of the
  reference (``true_hotwords`` of our bias lists), so this is also called
  **R-WER** (rare-word WER).

Japanese is written without spaces, so "words" come from a morphological
analyzer (:class:`src.utils.japanese.JapaneseTokenizer`: fugashi → SudachiPy →
pykakasi → char). A biasing word may still be split into several morphemes
(compound nouns, a different dictionary when the bias list was built, …), so
runs of consecutive tokens that spell a biasing word are merged back into one
token. A biasing word therefore counts as exactly one word, as an English rare
word does. Merging only happens on analyzer boundaries, so a short biasing word
is never carved out of a longer word (``京都`` inside ``東京都`` is left alone).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

from src.utils.compute_wer import normalize_japanese

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alignment (port of EditDistance in is21_deep_bias/score.py)
# ---------------------------------------------------------------------------
class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


_SUB_COST = 4
_INDEL_COST = 3

# (code, ref index or None, hyp index or None)
AlignedPair = Tuple[Code, Optional[int], Optional[int]]


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[AlignedPair]:
    """Levenshtein-align ``ref`` and ``hyp`` with the is21_deep_bias costs.

    Ties are broken as in the reference (diagonal, then insertion, then
    deletion, each only if strictly cheaper), so the alignment — and hence the
    U-WER / B-WER split — is identical. Unlike the reference, an empty ref
    *and* hyp is allowed and yields an empty alignment.
    """
    n, m = len(ref), len(hyp)
    diag, ins, dele = 0, 1, 2
    score = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[diag] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + _INDEL_COST
        back[0][j] = ins
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + _INDEL_COST
        back[i][0] = dele
        for j in range(1, m + 1):
            best = score[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else _SUB_COST)
            move = diag
            cand = score[i][j - 1] + _INDEL_COST
            if cand < best:
                best, move = cand, ins
            cand = score[i - 1][j] + _INDEL_COST
            if cand < best:
                best, move = cand, dele
            score[i][j] = best
            back[i][j] = move

    pairs: List[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == dele:
            i -= 1
            pairs.append((Code.deletion, i, None))
        elif move == ins:
            j -= 1
            pairs.append((Code.insertion, None, j))
        else:
            i, j = i - 1, j - 1
            code = Code.match if ref[i] == hyp[j] else Code.substitution
            pairs.append((code, i, j))
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Error counting
# ---------------------------------------------------------------------------
@dataclass
class ErrorCounts:
    """Edit counts of one metric (WER, U-WER or B-WER)."""

    ref_words: int = 0
    subs: int = 0
    ins: int = 0
    dels: int = 0

    @property
    def errors(self) -> int:
        return self.subs + self.ins + self.dels

    @property
    def rate(self) -> Optional[float]:
        """Error rate as a fraction; ``None`` when there are no reference words."""
        return self.errors / self.ref_words if self.ref_words else None

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.ref_words + other.ref_words,
            self.subs + other.subs,
            self.ins + other.ins,
            self.dels + other.dels,
        )

    def to_dict(self) -> dict:
        return {
            "error_rate": self.rate,
            "ref_words": self.ref_words,
            "subs": self.subs,
            "ins": self.ins,
            "dels": self.dels,
        }

    def __str__(self) -> str:
        rate = "n/a" if self.rate is None else f"{self.rate * 100:.2f}%"
        return (
            f"{rate} (ref_words={self.ref_words}, subs={self.subs}, "
            f"ins={self.ins}, dels={self.dels})"
        )


@dataclass
class BiasWERResult:
    """WER / U-WER / B-WER of a set of utterances (counts add up across sets)."""

    wer: ErrorCounts = field(default_factory=ErrorCounts)
    u_wer: ErrorCounts = field(default_factory=ErrorCounts)
    b_wer: ErrorCounts = field(default_factory=ErrorCounts)
    num_utts: int = 0
    # Biasing words absent from the tokenised reference (should be 0; >0 means
    # the segmentation disagrees with how the bias list was built).
    unmatched_biasing_words: int = 0
    word_splitter: str = ""

    def __add__(self, other: "BiasWERResult") -> "BiasWERResult":
        return BiasWERResult(
            self.wer + other.wer,
            self.u_wer + other.u_wer,
            self.b_wer + other.b_wer,
            self.num_utts + other.num_utts,
            self.unmatched_biasing_words + other.unmatched_biasing_words,
            self.word_splitter or other.word_splitter,
        )

    def to_dict(self) -> dict:
        return {
            "wer": self.wer.to_dict(),
            "u_wer": self.u_wer.to_dict(),
            "b_wer": self.b_wer.to_dict(),
            "num_utts": self.num_utts,
            "unmatched_biasing_words": self.unmatched_biasing_words,
            "word_splitter": self.word_splitter,
        }

    def __str__(self) -> str:
        return (
            f"WER: {self.wer}\nU-WER: {self.u_wer}\nB-WER (R-WER): {self.b_wer}"
        )


def count_errors(
    alignment: Sequence[AlignedPair],
    ref: Sequence[str],
    hyp: Sequence[str],
    biasing_words: Set[str],
    result: BiasWERResult,
) -> None:
    """Attribute every edit of ``alignment`` to WER and to U-WER or B-WER."""
    for code, ri, hi in alignment:
        if code == Code.insertion:
            biased = hyp[hi] in biasing_words
        else:
            biased = ref[ri] in biasing_words
        for counts in (result.wer, result.b_wer if biased else result.u_wer):
            if code != Code.insertion:
                counts.ref_words += 1
            if code == Code.substitution:
                counts.subs += 1
            elif code == Code.deletion:
                counts.dels += 1
            elif code == Code.insertion:
                counts.ins += 1


# ---------------------------------------------------------------------------
# Word splitting
# ---------------------------------------------------------------------------
def merge_biasing_words(
    tokens: Sequence[str], biasing_words: Set[str], joiner: str = ""
) -> List[str]:
    """Merge runs of consecutive tokens that spell a biasing word into one token.

    Greedy left to right, longest match first. Spans start and end on token
    boundaries only, so nothing is carved out of the middle of a token.
    """
    if not tokens or not biasing_words:
        return list(tokens)
    max_len = max(len(w) for w in biasing_words)
    out: List[str] = []
    i, n = 0, len(tokens)
    while i < n:
        end, span = i + 1, ""
        for j in range(i, n):
            span = tokens[j] if j == i else span + joiner + tokens[j]
            if len(span) > max_len:
                break
            if span in biasing_words:
                end = j + 1
        out.append(joiner.join(tokens[i:end]))
        i = end
    return out


class WordSplitter:
    """``text -> words``: whitespace for English, morphemes for Japanese.

    Args:
        language: ``"en"`` or ``"ja"``.
        backend: Japanese analyzer — ``"auto"`` (fugashi → sudachipy → pykakasi
            → char), or one of those names. Ignored for English.
    """

    def __init__(self, language: str = "en", backend: str = "auto") -> None:
        self.language = language
        self.joiner = "" if language == "ja" else " "
        self._tokenizer = None
        if language == "ja":
            from src.utils.japanese import JapaneseTokenizer

            self._tokenizer = JapaneseTokenizer(backend)
            self.name = self._tokenizer.backend_name
            if self.name == "char":
                logger.warning(
                    "No Japanese morphological analyzer available: words are single "
                    "characters, so WER / U-WER / B-WER degrade to character level. "
                    "Install fugashi + unidic-lite (or sudachipy + sudachidict-core)."
                )
        else:
            self.name = "whitespace"

    def __call__(self, text: str, biasing_words: Set[str] = frozenset()) -> List[str]:
        if self._tokenizer is not None:
            tokens = [
                t.surface for t in self._tokenizer.segment(text) if t.surface.strip()
            ]
        else:
            tokens = text.split()
        return merge_biasing_words(tokens, biasing_words, self.joiner)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
class BiasWERScorer:
    """Compute WER / U-WER / B-WER from raw hypotheses and references.

    Args:
        language: ``"en"`` (whitespace words) or ``"ja"`` (analyzer words).
        normalizer: text normaliser applied to hyps, refs *and* biasing words.
            Defaults to :func:`normalize_japanese` for ``ja`` and to identity
            for ``en`` (pass Whisper's ``BasicTextNormalizer`` there to match
            the WER of the inference script).
        ja_backend: Japanese analyzer, see :class:`WordSplitter`.
    """

    def __init__(
        self,
        language: str = "en",
        normalizer: Optional[Callable[[str], str]] = None,
        ja_backend: str = "auto",
    ) -> None:
        self.language = language
        if normalizer is None:
            normalizer = normalize_japanese if language == "ja" else (lambda s: s)
        self.normalize = normalizer
        self.split = WordSplitter(language, ja_backend)

    def normalize_biasing_words(self, words: Iterable[str]) -> Set[str]:
        out = set()
        for w in words:
            w = self.split.joiner.join(self.normalize(w.strip()).split())
            if w:
                out.add(w)
        return out

    def tokenize(
        self, hyp: str, ref: str, biasing_words: Iterable[str]
    ) -> Tuple[List[str], List[str], Set[str]]:
        """Normalise and split one utterance; returns ``(hyp, ref, biasing)``."""
        bias = self.normalize_biasing_words(biasing_words)
        hyp_tokens = self.split(self.normalize((hyp or "").strip()), bias)
        ref_tokens = self.split(self.normalize((ref or "").strip()), bias)
        return hyp_tokens, ref_tokens, bias

    def score(
        self,
        hyps: Sequence[str],
        refs: Sequence[str],
        biasing_words: Sequence[Iterable[str]],
    ) -> BiasWERResult:
        """Score parallel lists; ``biasing_words[i]`` are utterance i's rare words."""
        if not (len(hyps) == len(refs) == len(biasing_words)):
            raise ValueError(
                f"Length mismatch: {len(hyps)} hyps, {len(refs)} refs, "
                f"{len(biasing_words)} biasing-word lists"
            )
        result = BiasWERResult(word_splitter=self.split.name)
        for hyp, ref, words in zip(hyps, refs, biasing_words):
            hyp_tokens, ref_tokens, bias = self.tokenize(hyp, ref, words)
            count_errors(align(ref_tokens, hyp_tokens), ref_tokens, hyp_tokens, bias, result)
            result.num_utts += 1
            result.unmatched_biasing_words += len(bias - set(ref_tokens))
        if result.unmatched_biasing_words:
            logger.warning(
                "%d biasing word(s) not found as a word of their reference "
                "(segmentation differs from the bias list's); they are not "
                "counted in B-WER.",
                result.unmatched_biasing_words,
            )
        return result
