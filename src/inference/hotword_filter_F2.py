import Levenshtein
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class HotwordFilter:
    """
    Hotword filtering method F2: Common Word Removal + Threshold filtering.

    From the paper (Table 4):
      - Similarity Threshold: YES
      - Remove Common Words: YES
      - Match for Each Word: NO

    Algorithm:
      1. Build 2-gram index from the bias list
      2. Remove common words from CTC sentence (keep only rare words)
      3. Retrieve candidates via n-gram overlap with rare words
      4. Score each candidate against the whole filtered sentence
         (sliding window of matching length + word-level comparison)
      5. Select candidates with score >= threshold, up to word_num

    Args:
        common_words_path: Path to file with common (frequent) words, one per line
        n_gram: N-gram size for the index (default: 2)
        word_num: Maximum number of hotwords to keep (default: 15)
        probability_threshold: Minimum Levenshtein.ratio to accept (default: 0.9)
    """

    def __init__(
        self,
        common_words_path: Optional[str] = None,
        n_gram: int = 2,
        word_num: int = 15,
        probability_threshold: float = 0.9,
        per_word_scoring: bool = False,  # ignored in F2, kept for interface compat
    ):
        self.n_gram = n_gram
        self.word_num = word_num
        self.probability_threshold = probability_threshold
        self.top_k = word_num

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

    def score_candidates_whole_sentence(
        self, filtered_sentence: str, candidates: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Score each candidate against the whole filtered sentence.

        Uses both character-level sliding window and word-level comparison
        to find the best Levenshtein.ratio for each candidate.
        """
        scored = []
        sentence_lower = filtered_sentence.lower()
        words = sentence_lower.split()

        for candidate in candidates:
            cand_lower = candidate.lower()
            cand_len = len(cand_lower)
            if cand_len == 0:
                continue

            best_score = 0.0

            # Character-level sliding window
            for i in range(len(sentence_lower) - cand_len + 1):
                substr = sentence_lower[i : i + cand_len]
                score = Levenshtein.ratio(cand_lower, substr)
                best_score = max(best_score, score)

            # Word-level comparison
            for word in words:
                score = Levenshtein.ratio(cand_lower, word)
                best_score = max(best_score, score)

            scored.append((candidate, best_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def filter_hotwords(
        self, inference_sentence: str, hotwords: List[str]
    ) -> List[str]:
        """
        F2 filtering pipeline.

        1. Build n-gram index from bias list
        2. Remove common words from CTC sentence
        3. Retrieve candidates via n-gram overlap with rare words
        4. Score candidates against the whole filtered sentence
        5. Select by threshold, up to word_num
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

        # Step 4: Score candidates against the whole filtered sentence
        filtered_sentence = " ".join(rare_words)
        scored = self.score_candidates_whole_sentence(filtered_sentence, candidates)

        # Step 5: Select by threshold, up to word_num
        filtered = []
        for candidate, score in scored:
            if score >= self.probability_threshold and len(filtered) < self.word_num:
                filtered.append(candidate)

        logger.info(
            f"F2: Filtered {len(filtered)} hotwords from "
            f"{len(hotwords)} total (threshold={self.probability_threshold}, "
            f"max={self.word_num})"
        )
        return filtered
