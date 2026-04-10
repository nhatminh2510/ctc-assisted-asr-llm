import Levenshtein
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class HotwordFilter:
    """
    Hotword filtering method F1: Basic N-gram + Threshold filtering.

    From the paper (Table 4):
      - Similarity Threshold: YES
      - Remove Common Words: NO
      - Match for Each Word: NO

    Algorithm:
      1. Build 2-gram index from the bias list
      2. Use the FULL CTC sentence (no common word removal) to retrieve candidates
      3. Score each candidate against every word in the full sentence
      4. Select candidates with score >= threshold, up to word_num

    Args:
        common_words_path: Unused in F1 (kept for interface compatibility)
        n_gram: N-gram size for the index (default: 2)
        word_num: Maximum number of hotwords to keep (default: 5)
        probability_threshold: Minimum Levenshtein.ratio to accept (default: 0.95)
    """

    def __init__(
        self,
        common_words_path: Optional[str] = None,
        n_gram: int = 2,
        word_num: int = 5,
        probability_threshold: float = 0.95,
        per_word_scoring: bool = False,  # ignored in F1, kept for interface compat
    ):
        self.n_gram = n_gram
        self.word_num = word_num
        self.probability_threshold = probability_threshold
        self.top_k = word_num

    def build_ngram_index(self, hotwords: List[str]) -> Dict[str, Set[str]]:
        """Build n-gram index from hotword list."""
        ngram_index = defaultdict(set)
        for word in hotwords:
            word_lower = word.lower()
            for i in range(len(word_lower) - self.n_gram + 1):
                ngram = word_lower[i : i + self.n_gram]
                ngram_index[ngram].add(word_lower)
        return ngram_index

    def retrieve_candidates(
        self, words: List[str], ngram_index: Dict[str, Set[str]]
    ) -> Set[str]:
        """Retrieve candidate hotwords via n-gram overlap with sentence words."""
        candidates = set()
        for word in words:
            for i in range(len(word) - self.n_gram + 1):
                ngram = word[i : i + self.n_gram]
                if ngram in ngram_index:
                    candidates.update(ngram_index[ngram])
        return candidates

    def score_candidates(
        self, sentence_words: List[str], candidates: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Score each candidate against every word in the full sentence.

        For each candidate, the score is the maximum Levenshtein.ratio
        across all words in the sentence.
        """
        scored = []
        for candidate in candidates:
            cand_lower = candidate.lower()
            if not cand_lower:
                continue
            best_score = 0.0
            for word in sentence_words:
                score = Levenshtein.ratio(cand_lower, word)
                best_score = max(best_score, score)
            scored.append((candidate, best_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def filter_hotwords(
        self, inference_sentence: str, hotwords: List[str]
    ) -> List[str]:
        """
        F1 filtering pipeline.

        1. Build n-gram index from bias list
        2. Use FULL sentence (no common word removal) to retrieve candidates
        3. Score candidates against every word in the full sentence
        4. Select by threshold, up to word_num
        """
        if not hotwords:
            logger.warning("Empty hotwords list provided")
            return []
        if not inference_sentence:
            logger.warning("Empty inference sentence provided")
            return []

        # Step 1: Build n-gram index
        ngram_index = self.build_ngram_index(hotwords)

        # Step 2: Use full sentence words (NO common word removal)
        sentence_words = inference_sentence.lower().split()
        if not sentence_words:
            return []

        # Step 3: Retrieve candidates via n-gram overlap
        candidates = self.retrieve_candidates(sentence_words, ngram_index)
        if not candidates:
            logger.info("No candidates retrieved")
            return []

        # Step 4: Score candidates against full sentence words
        scored = self.score_candidates(sentence_words, candidates)

        # Step 5: Select by threshold, up to word_num
        filtered = []
        for candidate, score in scored:
            if score >= self.probability_threshold and len(filtered) < self.word_num:
                filtered.append(candidate)

        logger.info(
            f"F1: Filtered {len(filtered)} hotwords from "
            f"{len(hotwords)} total (threshold={self.probability_threshold}, "
            f"max={self.word_num})"
        )
        return filtered
