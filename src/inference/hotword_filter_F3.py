import Levenshtein
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class HotwordFilter:
    """
    Hotword filtering method F3: Per-Word Best-Match filtering (adopted method).

    From the paper (Table 4):
      - Similarity Threshold: NO
      - Remove Common Words: YES
      - Match for Each Word: YES

    Algorithm:
      1. Build 2-gram index from the bias list
      2. Remove common words from CTC sentence (keep only rare words)
      3. Retrieve candidates via n-gram overlap with rare words
      4. For EACH rare word, find the single best-matching candidate
         (top-K per word, default K=1)
      5. Collect the union of all per-word best matches (no threshold)

    This ensures every rare word in the CTC output gets its own best-matching
    hotword, providing better coverage than F1/F2.

    Args:
        common_words_path: Path to file with common (frequent) words, one per line
        n_gram: N-gram size for the index (default: 2)
        word_num: Unused in F3 (kept for interface compatibility)
        probability_threshold: Unused in F3 (kept for interface compatibility)
        first: Number of top matches per rare word (default: 1)
    """

    def __init__(
        self,
        common_words_path: Optional[str] = None,
        n_gram: int = 2,
        word_num: int = 15,  # kept for interface compat, not used in selection
        probability_threshold: float = 0.9,  # kept for interface compat, not used
        per_word_scoring: bool = True,  # ignored in F3, kept for interface compat
        first: int = 1,
    ):
        self.n_gram = n_gram
        self.word_num = word_num
        self.top_k = word_num
        self.first = first  # number of top matches per rare word

        # Load common words
        self.common_words: Set[str] = set()
        if common_words_path:
            try:
                with open(common_words_path, "r", encoding="utf-8") as f:
                    self.common_words = set(line.strip().lower() for line in f)
                logger.info(f"Loaded {len(self.common_words)} common words")
            except FileNotFoundError:
                logger.warning(f"Common words file not found: {common_words_path}")

        if not self.common_words:
            self.common_words = self._get_default_common_words()
            logger.info(f"Using default {len(self.common_words)} common words")

    def _get_default_common_words(self) -> Set[str]:
        """Default common words list (top ~100 English words)."""
        return set([
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no",
            "just", "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us",
        ])

    def build_ngram_index(self, hotwords: List[str]) -> Dict[str, Set[str]]:
        """Build n-gram index from hotword list."""
        ngram_index = defaultdict(set)
        for word in hotwords:
            word_lower = word.lower()
            for i in range(len(word_lower) - self.n_gram + 1):
                ngram = word_lower[i : i + self.n_gram]
                ngram_index[ngram].add(word_lower)
        return ngram_index

    def remove_common_words(self, sentence: str) -> List[str]:
        """Remove common words from the sentence, keeping only rare words."""
        words = sentence.lower().split()
        return [w for w in words if w not in self.common_words]

    def retrieve_candidates(
        self, rare_words: List[str], ngram_index: Dict[str, Set[str]]
    ) -> Set[str]:
        """Retrieve candidate hotwords via n-gram overlap with rare words."""
        candidates = set()
        for word in rare_words:
            for i in range(len(word) - self.n_gram + 1):
                ngram = word[i : i + self.n_gram]
                if ngram in ngram_index:
                    candidates.update(ngram_index[ngram])
        return candidates

    def score_candidates_per_word(
        self, rare_words: List[str], candidates: Set[str]
    ) -> List[str]:
        """
        For each rare word, find the top-K best-matching candidates.

        Matches the reference code's score_candidates_for_each_word():
        for each word in the filtered sentence, score all candidates,
        pick the top-self.first candidates, and collect their keys.

        No threshold is applied — every rare word contributes its best match.
        """
        selected = []
        candidates_list = list(candidates)

        for word in rare_words:
            word_lower = word.lower()
            scores = {}
            for candidate in candidates_list:
                score = Levenshtein.ratio(word_lower, candidate.lower())
                scores[candidate] = score

            sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            top_matches = sorted_items[: self.first]
            selected.extend(item[0] for item in top_matches)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w in selected:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        return unique

    def filter_hotwords(
        self, inference_sentence: str, hotwords: List[str]
    ) -> List[str]:
        """
        F3 filtering pipeline.

        1. Build n-gram index from bias list
        2. Remove common words from CTC sentence
        3. Retrieve candidates via n-gram overlap with rare words
        4. For each rare word, pick the top-1 best-matching candidate
        5. Return union of all per-word matches (no threshold)
        """
        if not hotwords:
            logger.warning("Empty hotwords list provided")
            return []
        if not inference_sentence:
            logger.warning("Empty inference sentence provided")
            return []

        # Step 1: Build n-gram index
        ngram_index = self.build_ngram_index(hotwords)

        # Step 2: Remove common words
        rare_words = self.remove_common_words(inference_sentence)
        if not rare_words:
            logger.info("No rare words found in inference sentence")
            return []

        # Step 3: Retrieve candidates via n-gram overlap
        candidates = self.retrieve_candidates(rare_words, ngram_index)
        if not candidates:
            logger.info("No candidates retrieved")
            return []

        # Step 4: Per-word best-match (no threshold)
        filtered = self.score_candidates_per_word(rare_words, candidates)

        logger.info(
            f"F3: Filtered {len(filtered)} hotwords from "
            f"{len(hotwords)} total (first={self.first}, per-word matching)"
        )
        return filtered
