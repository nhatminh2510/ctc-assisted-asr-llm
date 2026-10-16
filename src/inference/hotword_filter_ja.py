"""Japanese hotword filtering (method ``ja``).

This is the Japanese counterpart of ``hotword_filter_F1/F2/F3``. It keeps the
same public interface — :meth:`HotwordFilter.filter_hotwords` — so it plugs into
the existing inference pipeline, but it is redesigned around the three problems
that break the character/whitespace-based English filters:

* **No word boundaries.** The CTC hypothesis and the bias list are segmented
  with a morphological analyzer (see :mod:`src.utils.japanese`) instead of
  ``str.split()``, and hotwords are matched as *fuzzy substrings* of the whole
  hypothesis so recall survives an unsegmented kana run.

* **Polyphony (多音読み).** Retrieval and scoring are driven by the
  *reading* (context-resolved katakana), so 生 written in a hotword matches the
  reading the analyzer assigns it in the sentence, regardless of surface kanji.

* **Homophones (同音異字).** Candidates are retrieved through a reading n-gram
  index, which gives high recall for same-reading/different-kanji words even
  when the CTC model emits pure kana. Surface similarity is folded into the
  score to keep the correct kanji on top, and — because a kana hypothesis
  cannot disambiguate 橋/箸/端 on its own — all same-reading homophones can be
  emitted together so the LLM makes the final choice from the audio.

Algorithm (per utterance)::

    1. Build reading + surface n-gram indices from the bias list.
    2. Segment the CTC hypothesis and, if it contains kanji, drop
       function/common words (a kana-only run is kept whole), then
       concatenate the remaining tokens into reading/surface strings.
    3. Retrieve candidates via reading and surface n-gram overlap.
    4. Score each candidate = reading_similarity + surface_bonus * surface_sim,
       where each similarity is the best sliding-window match.
    5. Keep candidates above the threshold (or the top-`first`), then append
       same-reading homophones of the winners.
    6. De-duplicate, preserving order, and cap at `word_num`.

The selected hotwords are returned as **surface** strings (what should appear in
the transcript). :meth:`filter_hotwords_with_readings` additionally returns the
normalised reading of each, which the prompt builder uses to annotate them,
e.g. ``橋（ハシ）``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import Levenshtein

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.japanese import (
    JapaneseTokenizer,
    Token,
    to_katakana,
    contains_kanji,
)

logger = logging.getLogger(__name__)


# Coarse POS tags treated as function words and removed before matching.
# Covers UniDic (補助記号/助詞…) and IPADIC (記号) style tags.
_FUNCTION_POS = {
    "助詞",
    "助動詞",
    "接続詞",
    "感動詞",
    "フィラー",
    "記号",
    "補助記号",
    "空白",
}

# A small default stop set used when the backend has no POS (e.g. pykakasi/char)
# or when no common-words file is supplied. Kept intentionally short — the
# reading index makes over-removal the bigger risk than under-removal.
_DEFAULT_STOPWORDS = {
    "は", "が", "を", "に", "へ", "と", "で", "や", "の", "も", "から", "まで",
    "より", "ね", "よ", "な", "か", "です", "ます", "だ", "である", "た", "て",
    "し", "する", "した", "して", "いる", "ある", "なる", "れる", "られる",
    "この", "その", "あの", "どの", "これ", "それ", "あれ", "こと", "もの",
    "ため", "など", "さん", "ちゃん", "くん",
}


@dataclass
class _HotwordEntry:
    surface: str
    reading: str
    reading_ngrams: Set[str] = field(default_factory=set)
    surface_ngrams: Set[str] = field(default_factory=set)


class HotwordFilter:
    """Japanese hotword filter.

    Args:
        common_words_path: Optional path to a file of common words (one per
            line; surface or reading). These are removed from the hypothesis in
            addition to POS-based / default stopword removal — only for
            hypotheses that contain kanji (see :meth:`extract_content_tokens`).
        n_gram: N-gram size for the reading/surface indices (default: 2).
        word_num: Maximum number of hotwords to keep (default: 15).
        probability_threshold: Minimum score to accept a candidate. Acts as a
            precision floor (default 0.5). Set to 0 for maximum recall
            (F3-style, no threshold).
        per_word_scoring: Selection strategy. False (default) keeps every
            candidate scoring above ``probability_threshold`` up to
            ``word_num`` (F2-style, best recall — recommended for Japanese
            because kana CTC output is hard to segment). True keeps only the
            top-``first`` highest-scoring hotwords (higher precision).
        first: Number of hotwords kept when ``per_word_scoring`` is True
            (default: 1).
        reading_weight: Weight of the reading-similarity term (default: 1.0).
        surface_bonus: Weight of the surface-similarity term, used to break ties
            among homophones when the hypothesis contains kanji (default: 0.3).
        include_homophones: If True, also emit bias-list words that share the
            exact reading of a selected hotword (default: True). This is what
            lets a kana-only CTC hypothesis surface 橋/箸/端 together.
        max_homophones: Cap on homophones added per selected hotword
            (default: 3).
        tokenizer_backend: Passed to :class:`JapaneseTokenizer`
            (``"auto"`` by default).
    """

    def __init__(
        self,
        common_words_path: Optional[str] = None,
        n_gram: int = 2,
        word_num: int = 15,
        probability_threshold: float = 0.5,
        per_word_scoring: bool = False,
        first: int = 1,
        reading_weight: float = 1.0,
        surface_bonus: float = 0.3,
        include_homophones: bool = True,
        max_homophones: int = 3,
        tokenizer_backend: str = "auto",
    ) -> None:
        self.n_gram = max(1, n_gram)
        self.word_num = word_num
        self.top_k = word_num
        self.probability_threshold = probability_threshold
        self.per_word_scoring = per_word_scoring
        self.first = first
        self.reading_weight = reading_weight
        self.surface_bonus = surface_bonus
        self.include_homophones = include_homophones
        self.max_homophones = max_homophones

        self.tokenizer = JapaneseTokenizer(backend=tokenizer_backend)

        # Common words are matched by both surface and normalised reading.
        self.common_surfaces: Set[str] = set()
        self.common_readings: Set[str] = set()
        if common_words_path:
            try:
                with open(common_words_path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if not w:
                            continue
                        self.common_surfaces.add(w)
                        self.common_readings.add(to_katakana(w))
                logger.info(
                    "Loaded %d Japanese common words from %s",
                    len(self.common_surfaces),
                    common_words_path,
                )
            except FileNotFoundError:
                logger.warning(
                    "Common words file not found: %s (using defaults)",
                    common_words_path,
                )

        if not self.common_surfaces:
            self.common_surfaces = set(_DEFAULT_STOPWORDS)
            self.common_readings = {to_katakana(w) for w in _DEFAULT_STOPWORDS}

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _char_ngrams(self, text: str) -> Set[str]:
        """Character n-grams of *text* (whole string if shorter than n)."""
        if not text:
            return set()
        if len(text) < self.n_gram:
            return {text}
        return {text[i : i + self.n_gram] for i in range(len(text) - self.n_gram + 1)}

    def _build_entries(
        self, hotwords: List[str]
    ) -> Tuple[
        List[_HotwordEntry],
        Dict[str, Set[int]],
        Dict[str, Set[int]],
        Dict[str, Set[int]],
    ]:
        """Build hotword entries plus reading/surface n-gram and reading-exact
        indices. Indices map an n-gram (or exact reading) to hotword indices."""
        entries: List[_HotwordEntry] = []
        reading_index: Dict[str, Set[int]] = defaultdict(set)
        surface_index: Dict[str, Set[int]] = defaultdict(set)
        reading_exact: Dict[str, Set[int]] = defaultdict(set)

        for hw in hotwords:
            if not hw:
                continue
            reading = to_katakana(hw)
            r_ngrams = self._char_ngrams(reading)
            s_ngrams = self._char_ngrams(hw)
            entries.append(_HotwordEntry(hw, reading, r_ngrams, s_ngrams))
            entry_idx = len(entries) - 1
            for g in r_ngrams:
                reading_index[g].add(entry_idx)
            for g in s_ngrams:
                surface_index[g].add(entry_idx)
            if reading:
                reading_exact[reading].add(entry_idx)

        return entries, reading_index, surface_index, reading_exact

    # ------------------------------------------------------------------
    # Hypothesis pre-processing
    # ------------------------------------------------------------------
    def _is_function_word(self, token: Token) -> bool:
        if token.pos is not None and token.pos in _FUNCTION_POS:
            return True
        if token.surface in self.common_surfaces:
            return True
        if token.reading and token.reading in self.common_readings:
            return True
        return False

    def extract_content_tokens(self, sentence: str) -> List[Token]:
        """Segment *sentence* and keep only content (non-function) tokens.

        Function/common-word removal only runs when *sentence* contains kanji
        (and the backend is not the character-level fallback). A kana-only
        hypothesis — the usual kana-CTC output — is one unsegmented run that
        the analyzer splits mid-word (パターン → タパ/ターン, ツカム → ツカ/ム), so
        its POS tags are unreliable and dropping the "function" or common
        pieces strips syllables out of real words: 掴む, 視察, 避難 … vanish.
        The same happens with the character fallback, where single-kana
        stopwords (は/し/も/と…) would erase はしもと. The sliding-window match
        tolerates the extra particles, so the run is kept whole. On CSJ
        eval1–3 this raises hotword recall from ~75% to ~91% for about one
        point of precision.
        """
        tokens = self.tokenizer.segment(sentence)
        if self.tokenizer.backend_name == "char" or not contains_kanji(sentence):
            return tokens
        return [t for t in tokens if not self._is_function_word(t)]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _best_window_ratio(needle: str, haystack: str) -> float:
        """Best ``Levenshtein.ratio`` of *needle* against any sub-span of
        *haystack*.

        This is the key to the no-boundary problem: a kana-CTC hypothesis is
        often one long unsegmented run, so we search for the hotword reading as
        a fuzzy substring instead of relying on word segmentation. Windows of
        length ``len(needle) ± 1`` are tried to tolerate a single ins/deletion.
        """
        n, h = len(needle), len(haystack)
        if n == 0 or h == 0:
            return 0.0
        if n >= h:
            return Levenshtein.ratio(needle, haystack)
        best = 0.0
        for win in (n, n + 1, n - 1):
            if win <= 0 or win > h:
                continue
            for i in range(h - win + 1):
                r = Levenshtein.ratio(needle, haystack[i : i + win])
                if r > best:
                    best = r
                    if best == 1.0:
                        return best
        return best

    def _score(self, entry: _HotwordEntry, hyp_reading: str, hyp_surface: str) -> float:
        """Combined reading + surface similarity of a hotword vs the hypothesis.

        Reading similarity does the heavy lifting (handles polyphony and lets a
        kana hypothesis reach kanji hotwords). Surface similarity is only added
        when *both* sides carry kanji, so it disambiguates homophones when the
        CTC model happens to emit kanji, without penalising the common kana-only
        case.
        """
        reading_sim = self._best_window_ratio(entry.reading, hyp_reading)
        surface_sim = 0.0
        if contains_kanji(entry.surface) and contains_kanji(hyp_surface):
            surface_sim = self._best_window_ratio(entry.surface, hyp_surface)
        return self.reading_weight * reading_sim + self.surface_bonus * surface_sim

    def _retrieve_from_strings(
        self,
        hyp_reading: str,
        hyp_surface: str,
        reading_index: Dict[str, Set[int]],
        surface_index: Dict[str, Set[int]],
    ) -> Set[int]:
        """Candidate hotwords sharing a reading or surface n-gram with the
        (segmentation-independent) hypothesis strings."""
        cand: Set[int] = set()
        for g in self._char_ngrams(hyp_reading):
            if g in reading_index:
                cand |= reading_index[g]
        for g in self._char_ngrams(hyp_surface):
            if g in surface_index:
                cand |= surface_index[g]
        return cand

    def _build_hypothesis_strings(self, sentence: str) -> Tuple[str, str]:
        """Return ``(reading, surface)`` strings for matching.

        Function/common words are dropped via segmentation; when the segmenter
        cannot split a run (e.g. a long kana CTC output under the pykakasi/char
        fallback) we fall back to the whole sentence so recall is preserved.
        """
        content_tokens = self.extract_content_tokens(sentence)
        hyp_reading = "".join(t.reading for t in content_tokens)
        hyp_surface = "".join(t.surface for t in content_tokens)
        if not hyp_reading:
            hyp_reading = to_katakana(sentence)
            hyp_surface = sentence
        return hyp_reading, hyp_surface

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def filter_hotwords_with_readings(
        self, inference_sentence: str, hotwords: List[str]
    ) -> List[Tuple[str, str]]:
        """Filter and return ``[(surface, reading), ...]``."""
        if not hotwords:
            logger.warning("Empty hotwords list provided")
            return []
        if not inference_sentence:
            logger.warning("Empty inference sentence provided")
            return []

        entries, reading_index, surface_index, reading_exact = self._build_entries(
            hotwords
        )
        if not entries:
            return []

        hyp_reading, hyp_surface = self._build_hypothesis_strings(inference_sentence)
        if not hyp_reading:
            logger.info("Empty hypothesis reading after preprocessing")
            return []

        cand_ids = self._retrieve_from_strings(
            hyp_reading, hyp_surface, reading_index, surface_index
        )
        if not cand_ids:
            logger.info("No candidates retrieved")
            return []

        scored = sorted(
            (
                (cid, self._score(entries[cid], hyp_reading, hyp_surface))
                for cid in cand_ids
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        selected: List[int] = []
        seen: Set[int] = set()

        def _add(idx: int) -> None:
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)

        if self.per_word_scoring:
            # Keep only the top-`first` best matches (subject to the floor).
            kept = 0
            for cid, score in scored:
                if score < self.probability_threshold:
                    break
                _add(cid)
                self._maybe_add_homophones(cid, entries, reading_exact, _add)
                kept += 1
                if kept >= self.first or len(selected) >= self.word_num:
                    break
        else:
            # Keep everything above the threshold, up to word_num.
            for cid, score in scored:
                if score < self.probability_threshold:
                    break
                _add(cid)
                self._maybe_add_homophones(cid, entries, reading_exact, _add)
                if len(selected) >= self.word_num:
                    break

        result = [
            (entries[i].surface, entries[i].reading) for i in selected[: self.word_num]
        ]
        logger.info(
            "ja: Filtered %d hotwords from %d total "
            "(backend=%s, per_word=%s, homophones=%s)",
            len(result),
            len(hotwords),
            self.tokenizer.backend_name,
            self.per_word_scoring,
            self.include_homophones,
        )
        return result

    def _maybe_add_homophones(
        self,
        chosen_idx: int,
        entries: List[_HotwordEntry],
        reading_exact: Dict[str, Set[int]],
        add_fn,
    ) -> None:
        """Append same-reading homophones of a chosen hotword (同音異字)."""
        if not self.include_homophones:
            return
        reading = entries[chosen_idx].reading
        if not reading:
            return
        added = 0
        for cid in sorted(reading_exact.get(reading, set())):
            if cid == chosen_idx:
                continue
            add_fn(cid)
            added += 1
            if added >= self.max_homophones:
                break

    def filter_hotwords(self, inference_sentence: str, hotwords: List[str]) -> List[str]:
        """Filter and return surface strings (interface-compatible with F1–F3)."""
        return [
            surface
            for surface, _ in self.filter_hotwords_with_readings(
                inference_sentence, hotwords
            )
        ]
