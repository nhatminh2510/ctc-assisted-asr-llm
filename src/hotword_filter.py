import torch
import Levenshtein
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import logging
import string
import numpy as np

logger = logging.getLogger(__name__)

class HotwordFilter:
    """
    Filter hotwords from a large bias list using CTC pseudo-labels.

    Implements the filtering algorithm from the paper:
      1. Build bigram (n-gram) index from bias list
      2. Remove common words from CTC output
      3. Retrieve candidate hotwords via n-gram overlap
      4. Score candidates using Levenshtein.ratio
      5. Select top-K words above probability_threshold

    Two scoring strategies (matching reference code):
      - Whole-sentence scoring: score each candidate against the full CTC sentence
      - Per-word scoring: score each candidate against individual rare words

    Args:
        common_words_path: Path to file with common (frequent) words, one per line
        n_gram: N-gram size for the index (default: 2 = bigram)
        word_num: Maximum number of hotwords to keep (default: 15)
        probability_threshold: Minimum Levenshtein.ratio to accept (default: 0.9)
        per_word_scoring: If True, use per-word scoring; else whole-sentence (default: True)

    Reference: https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/contextual_asr
    """

    def __init__(
        self,
        common_words_path: Optional[str] = None,
        n_gram: int = 2,
        word_num: int = 15,
        probability_threshold: float = 0.9,
        per_word_scoring: bool = True,
    ):
        self.n_gram = n_gram
        self.word_num = word_num
        self.probability_threshold = probability_threshold
        self.per_word_scoring = per_word_scoring

        # Backward compat alias
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
        """Định nghĩa danh sách từ phổ biến mặc định."""
        return set(
            [
                "the",
                "be",
                "to",
                "of",
                "and",
                "a",
                "in",
                "that",
                "have",
                "i",
                "it",
                "for",
                "not",
                "on",
                "with",
                "he",
                "as",
                "you",
                "do",
                "at",
                "this",
                "but",
                "his",
                "by",
                "from",
                "they",
                "we",
                "say",
                "her",
                "she",
                "or",
                "an",
                "will",
                "my",
                "one",
                "all",
                "would",
                "there",
                "their",
                "what",
                "so",
                "up",
                "out",
                "if",
                "about",
                "who",
                "get",
                "which",
                "go",
                "me",
                "when",
                "make",
                "can",
                "like",
                "time",
                "no",
                "just",
                "him",
                "know",
                "take",
                "people",
                "into",
                "year",
                "your",
                "good",
                "some",
                "could",
                "them",
                "see",
                "other",
                "than",
                "then",
                "now",
                "look",
                "only",
                "come",
                "its",
                "over",
                "think",
                "also",
                "back",
                "after",
                "use",
                "two",
                "how",
                "our",
                "work",
                "first",
                "well",
                "way",
                "even",
                "new",
                "want",
                "because",
                "any",
                "these",
                "give",
                "day",
                "most",
                "us",
            ]
        )

    def build_ngram_index(self, hotwords: List[str]) -> Dict[str, Set[str]]:
        """
        Xây dựng từ điển chỉ mục n-gram từ danh sách hotwords.

        Args:
            hotwords: Danh sách các hotwords

        Returns:
            ngram_index: Dictionary mapping n-grams đến các hotwords chứa chúng

        Ví dụ:
            hotwords = ['bertie', 'albert']
            n_gram = 2
            -> {'be': {'bertie'}, 'er': {'bertie'}, 'rt': {'bertie', 'albert'},
                'ti': {'bertie'}, 'ie': {'bertie'}, 'al': {'albert'}, 'lb': {'albert'}, ...}
        """
        ngram_index = defaultdict(set)

        for word in hotwords:
            word_lower = word.lower()
            # Tạo tất cả các n-grams của word
            for i in range(len(word_lower) - self.n_gram + 1):
                ngram = word_lower[i : i + self.n_gram]
                ngram_index[ngram].add(word_lower)

        return ngram_index

    def remove_common_words(self, sentence: str) -> List[str]:
        """
        Loại bỏ các từ phổ biến từ câu.

        Args:
            sentence: Câu phiên âm sơ bộ

        Returns:
            rare_words: Danh sách các từ hiếm (không phổ biến)
        """
        words = sentence.lower().split()
        rare_words = [word for word in words if word not in self.common_words]
        return rare_words

    def retrieve_candidates(
        self, rare_words: List[str], ngram_index: Dict[str, Set[str]]
    ) -> Set[str]:
        """
        Retrieve candidate hotwords based on n-gram overlap with rare words.

        Args:
            rare_words: Rare words extracted from CTC output
            ngram_index: N-gram index built from the bias list

        Returns:
            candidates: Set of candidate hotwords
        """
        candidates = set()

        for word in rare_words:
            for i in range(len(word) - self.n_gram + 1):
                ngram = word[i : i + self.n_gram]
                if ngram in ngram_index:
                    candidates.update(ngram_index[ngram])

        return candidates

    def score_candidates_whole_sentence(
        self, ctc_sentence: str, candidates: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Score candidates against the whole CTC sentence using Levenshtein.ratio.

        For each candidate, find the best matching substring in the sentence.

        Args:
            ctc_sentence: Full CTC decoded sentence (lowercased)
            candidates: Set of candidate hotwords

        Returns:
            List of (candidate, score) tuples, sorted by score descending
        """
        scored = []
        sentence_lower = ctc_sentence.lower()

        for candidate in candidates:
            cand_lower = candidate.lower()
            # Slide a window over the sentence to find best match
            best_score = 0.0
            cand_len = len(cand_lower)

            if cand_len == 0:
                continue

            # Check word-level substrings of similar length
            words = sentence_lower.split()
            # Also check character-level windows for robustness
            for i in range(len(sentence_lower) - cand_len + 1):
                substr = sentence_lower[i : i + cand_len]
                score = Levenshtein.ratio(cand_lower, substr)
                best_score = max(best_score, score)

            # Also check each word directly
            for word in words:
                score = Levenshtein.ratio(cand_lower, word)
                best_score = max(best_score, score)

            scored.append((candidate, best_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def score_candidates_per_word(
        self, rare_words: List[str], candidates: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Score candidates against individual rare words (per-word strategy).

        For each rare word, find the best matching candidate.
        This matches the reference `score_candidates_for_each_word()`.

        Args:
            rare_words: Rare words from CTC output
            candidates: Set of candidate hotwords

        Returns:
            List of (candidate, score) tuples, sorted by score descending
        """
        # For each rare word, find best candidate match
        best_per_candidate: Dict[str, float] = {}

        for rare_word in rare_words:
            for candidate in candidates:
                score = Levenshtein.ratio(rare_word.lower(), candidate.lower())
                if candidate not in best_per_candidate:
                    best_per_candidate[candidate] = score
                else:
                    best_per_candidate[candidate] = max(
                        best_per_candidate[candidate], score
                    )

        scored = list(best_per_candidate.items())
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def match_hotwords(self, rare_words: List[str], candidates: Set[str]) -> List[str]:
        """
        Match and select the best hotwords using Levenshtein.ratio scoring.

        Applies probability_threshold and word_num limits.

        Args:
            rare_words: Rare words from CTC output
            candidates: Candidate hotwords from n-gram retrieval

        Returns:
            Filtered hotwords (max word_num items)
        """
        if not candidates:
            return []

        if self.per_word_scoring:
            scored = self.score_candidates_per_word(rare_words, candidates)
        else:
            ctc_sentence = " ".join(rare_words)
            scored = self.score_candidates_whole_sentence(ctc_sentence, candidates)

        # Filter by threshold and limit
        filtered = []
        for candidate, score in scored:
            if score >= self.probability_threshold and len(filtered) < self.word_num:
                filtered.append(candidate)

        return filtered

    def filter_hotwords(
        self, inference_sentence: str, hotwords: List[str]
    ) -> List[str]:
        """
        Main filtering method.

        Pipeline:
          1. Build n-gram index from bias list
          2. Remove common words from CTC sentence
          3. Retrieve candidates via n-gram overlap
          4. Score and select top hotwords

        Args:
            inference_sentence: CTC decoded text (pseudo-label)
            hotwords: Full bias list (true hotwords + distractors)

        Returns:
            Filtered hotwords (max word_num items above threshold)
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

        # Step 3: Retrieve candidates
        candidates = self.retrieve_candidates(rare_words, ngram_index)

        if not candidates:
            logger.info("No candidates retrieved")
            return []

        # Step 4: Score and select
        filtered_hotwords = self.match_hotwords(rare_words, candidates)

        logger.info(
            f"Filtered {len(filtered_hotwords)} hotwords from "
            f"{len(hotwords)} total (threshold={self.probability_threshold}, "
            f"max={self.word_num})"
        )

        return filtered_hotwords


def create_prompt_with_hotwords(
    filtered_hotwords: List[str],
    transcription: str = "",
) -> str:
    """
    Create prompt text with filtered hotwords for LLM.
    Matches reference Vicuna chat format: "USER: {prompt}\\n ASSISTANT:{answer}</s>"

    Args:
        filtered_hotwords: List of filtered hotwords
        transcription: Transcription text (if provided, EOS '</s>' is appended)

    Returns:
        prompt: Prompt text in Vicuna chat template format

    Examples:
        filtered_hotwords = ['bertie', 'alice']
        -> 'USER: Transcribe speech to text. Some hotwords might help. The hotwords are "bertie alice".\n ASSISTANT:'

        filtered_hotwords = ['bertie', 'alice'], transcription = "hello world"
        -> 'USER: ... The hotwords are "bertie alice".\n ASSISTANT:hello world</s>'
    """
    # Build answer part with EOS token
    answer = ""
    if transcription:
        answer = f"{transcription}</s>"

    # Build prompt with correct Vicuna chat format (matching reference)
    if not filtered_hotwords:
        prompt = f"USER: Transcribe speech to text.\n ASSISTANT:{answer}"
    else:
        hotwords_str = " ".join(filtered_hotwords)
        prompt = (
            f"USER: Transcribe speech to text. Some hotwords might help. "
            f'The hotwords are "{hotwords_str}".\n ASSISTANT:{answer}'
        )

    return prompt
