import torch
import editdistance
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import logging
import string
import numpy as np

logger = logging.getLogger(__name__)


class CTCDecoder:
    """
    CTC Decoder sử dụng thuật toán Viterbi để tìm phiên âm tối ưu nhất.

    Args:
        vocab (List[str]): Danh sách các ký tự trong từ vựng
        blank_id (int): ID của blank token trong CTC
    """

    def __init__(self, vocab: Optional[List[str]] = None, blank_id: int = 0):
        if vocab is None:
            # Tầo vocab mặc định: blank + space + a-z + apostrophe
            self.vocab = ["<blank>", " "] + list(string.ascii_lowercase) + ["'"]
        else:
            self.vocab = vocab

        self.blank_id = blank_id
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        logger.info(f"Initialized CTCDecoder with vocab size: {len(self.vocab)}")

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Giải mã CTC logits thành văn bản sử dụng greedy decoding.

        Args:
            logits: Tensor shape (sequence_length, vocab_size) - Log probabilities từ CTC

        Returns:
            decoded_text: String phiên âm đã giải mã
        """
        # Greedy decoding: lấy token có xác suất cao nhất tại mỗi timestep
        predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        # Loại bỏ các token trùng lặp và blank tokens
        decoded_chars = []
        prev_id = None

        for token_id in predicted_ids:
            # Bỏ qua blank token
            if token_id == self.blank_id:
                prev_id = None
                continue

            # Bỏ qua token trùng lặp liên tiếp
            if token_id == prev_id:
                continue

            # Thêm ký tự vào kết quả
            if token_id < len(self.vocab):
                decoded_chars.append(self.id_to_char[token_id])

            prev_id = token_id

        decoded_text = "".join(decoded_chars).strip()
        return decoded_text

    def viterbi_decode(self, logits: torch.Tensor) -> str:
        """
        Giải mã CTC logits thành văn bản sử dụng thuật toán Viterbi.

        Thuật toán Viterbi tìm chuỗi ký tự có xác suất cao nhất bằng cách:
        1. Duy trì bảng xác suất cho mỗi trạng thái tại mỗi bước thời gian
        2. Theo dõi đường dẫn ngược để tái tạo chuỗi tối ưu
        3. Loại bỏ blank tokens và các token trùng lặp liên tiếp

        Args:
            logits: Tensor shape (sequence_length, vocab_size) - Log probabilities từ CTC

        Returns:
            decoded_text: String phiên âm đã giải mã
        """
        # Chuyển logits thành numpy cho tính toán Viterbi
        if isinstance(logits, torch.Tensor):
            logits_np = logits.cpu().detach().numpy()
        else:
            logits_np = logits

        sequence_length, vocab_size = logits_np.shape

        # Khởi tạo bảng Viterbi
        # viterbi[t, k] = xác suất cao nhất của chuỗi kết thúc tại token k ở bước t
        viterbi = np.full((sequence_length, vocab_size), -np.inf)
        backpointer = np.zeros((sequence_length, vocab_size), dtype=np.int32)

        # Khởi tạo bước thời gian đầu tiên
        viterbi[0, :] = logits_np[0, :]

        # Điền bảng Viterbi từ trước đến sau
        for t in range(1, sequence_length):
            for k in range(vocab_size):
                # Xác suất chuyển đến token k từ bất kỳ token nào ở bước trước
                # Trong CTC, bất kỳ token nào cũng có thể chuyển đến bất kỳ token nào
                prev_probs = viterbi[t - 1, :] + logits_np[t, k]

                # Tìm token trước có xác suất cao nhất
                best_prev_token = np.argmax(prev_probs)
                viterbi[t, k] = prev_probs[best_prev_token]
                backpointer[t, k] = best_prev_token

        # Tìm token với xác suất cao nhất ở bước cuối cùng
        last_token = np.argmax(viterbi[sequence_length - 1, :])

        # Tái tạo chuỗi tối ưu bằng backtracking
        path = []
        current_token = last_token

        for t in range(sequence_length - 1, -1, -1):
            path.append(current_token)
            if t > 0:
                current_token = backpointer[t, current_token]

        path.reverse()

        # Post-processing: loại bỏ blank tokens và tokens trùng lặp liên tiếp
        decoded_chars = []
        prev_token = None

        for token_id in path:
            # Bỏ qua blank token
            if token_id == self.blank_id:
                prev_token = None
                continue

            # Bỏ qua token trùng lặp liên tiếp
            if token_id == prev_token:
                continue

            # Thêm ký tự vào kết quả
            if token_id < len(self.vocab):
                decoded_chars.append(self.id_to_char[token_id])

            prev_token = token_id

        decoded_text = "".join(decoded_chars).strip()
        return decoded_text

    def decode(self, logits: torch.Tensor) -> str:
        """
        Giải mã CTC logits thành văn bản sử dụng Viterbi algorithm.

        Args:
            logits: Tensor shape (sequence_length, vocab_size) - Log probabilities từ CTC

        Returns:
            decoded_text: String phiên âm đã giải mã
        """
        return self.viterbi_decode(logits)

    def batch_decode(self, logits_batch: torch.Tensor) -> List[str]:
        """
        Giải mã batch các CTC logits.

        Args:
            logits_batch: Tensor shape (batch_size, sequence_length, vocab_size)

        Returns:
            decoded_texts: List các phiên âm đã giải mã
        """
        return [self.decode(logits) for logits in logits_batch]


class HotwordFilter:
    """
    Thuật toán lọc hotwords để chọn ra các từ phù hợp nhất từ danh sách hotwords lớn.

    Args:
        common_words_path (str): Đường dẫn đến file chứa danh sách từ phổ biến
        n_gram (int): Kích thước n-gram cho index dictionary (mặc định: 2)
        top_k (int): Số hotwords tối đa được giữ lại (mặc định: 10)
    """

    def __init__(
        self, common_words_path: Optional[str] = None, n_gram: int = 2, top_k: int = 10
    ):
        self.n_gram = n_gram
        self.top_k = top_k

        # Load danh sách từ phổ biến
        self.common_words: Set[str] = set()
        if common_words_path:
            try:
                with open(common_words_path, "r", encoding="utf-8") as f:
                    self.common_words = set(line.strip().lower() for line in f)
                logger.info(f"Loaded {len(self.common_words)} common words")
            except FileNotFoundError:
                logger.warning(f"Common words file not found: {common_words_path}")

        # Nếu không có file, sử dụng một số từ phổ biến cơ bản
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
        Truy xuất các hotwords ứng viên dựa trên các từ hiếm.

        Args:
            rare_words: Danh sách các từ hiếm
            ngram_index: Từ điển chỉ mục n-gram

        Returns:
            candidates: Tập hợp các hotwords ứng viên
        """
        candidates = set()

        for word in rare_words:
            # Tạo tất cả n-grams của word
            for i in range(len(word) - self.n_gram + 1):
                ngram = word[i : i + self.n_gram]
                # Truy xuất hotwords chứa n-gram này
                if ngram in ngram_index:
                    candidates.update(ngram_index[ngram])

        return candidates

    def match_hotwords(self, rare_words: List[str], candidates: Set[str]) -> List[str]:
        """
        Đối sánh và lựa chọn hotwords phù hợp nhất cho mỗi từ hiếm.

        Args:
            rare_words: Danh sách các từ hiếm từ phiên âm sơ bộ
            candidates: Tập hợp các hotwords ứng viên

        Returns:
            matched_hotwords: Danh sách các hotwords được chọn (tối đa top_k)
        """
        matched_hotwords = set()
        candidates_list = list(candidates)

        # Với mỗi từ hiếm, tìm hotword có edit distance nhỏ nhất
        for rare_word in rare_words:
            if not candidates_list:
                continue

            # Tính edit distance đến tất cả các candidates
            distances = [
                (editdistance.eval(rare_word, candidate), candidate)
                for candidate in candidates_list
            ]

            # Chọn candidate có distance nhỏ nhất
            distances.sort(key=lambda x: x[0])
            best_match = distances[0][1]
            matched_hotwords.add(best_match)

        # Giới hạn số lượng hotwords trả về
        return list(matched_hotwords)[: self.top_k]

    def filter_hotwords(
        self, inference_sentence: str, hotwords: List[str]
    ) -> List[str]:
        """
        Phương thức chính để lọc hotwords.

        Args:
            inference_sentence: Câu phiên âm sơ bộ từ CTC decoder
            hotwords: Danh sách tất cả các hotwords (bao gồm nhiều từ gây nhiễu)

        Returns:
            filtered_hotwords: Danh sách hotwords đã được lọc (tối đa top_k)

        Thuật toán:
        1. Xây dựng từ điển n-gram index
        2. Loại bỏ từ phổ biến từ inference sentence
        3. Truy xuất candidates dựa trên từ hiếm
        4. Đối sánh và chọn hotwords tốt nhất
        """
        if not hotwords:
            logger.warning("Empty hotwords list provided")
            return []

        if not inference_sentence:
            logger.warning("Empty inference sentence provided")
            return []

        # Bước 1: Xây dựng n-gram index
        ngram_index = self.build_ngram_index(hotwords)

        # Bước 2: Loại bỏ từ phổ biến
        rare_words = self.remove_common_words(inference_sentence)

        if not rare_words:
            logger.info("No rare words found in inference sentence")
            return []

        # Bước 3: Truy xuất candidates
        candidates = self.retrieve_candidates(rare_words, ngram_index)

        if not candidates:
            logger.info("No candidates retrieved")
            return []

        # Bước 4: Đối sánh và chọn hotwords
        filtered_hotwords = self.match_hotwords(rare_words, candidates)

        logger.info(
            f"Filtered {len(filtered_hotwords)} hotwords from "
            f"{len(hotwords)} total hotwords"
        )

        return filtered_hotwords


def create_prompt_with_hotwords(
    filtered_hotwords: List[str],
    transcription: str = "",
) -> str:
    """
    Tạo prompt text chứa các filtered hotwords cho LLM.

    Args:
        filtered_hotwords: Danh sách hotwords đã được lọc
        transcription: Transcription text (nếu có, thêm EOS token '</s>' ở cuối)

    Returns:
        prompt: Prompt text theo định dạng

    Ví dụ:
        filtered_hotwords = ['bertie', 'alice']
        -> "USER: Transcribe speech to text. Some hotwords might help.
            The hotwords are bertie, alice. ASSISTANT:"
    """
    # Thêm EOS token vào cuối transcription để model học khi nào dừng
    if transcription:
        transcription = f"{transcription}</s>"

    if not filtered_hotwords:
        prompt = f"USER: Transcribe speech to text. ASSISTANT: {transcription}"
    else:
        hotwords_str = ", ".join(filtered_hotwords)
        prompt = (
            f"USER: Transcribe speech to text. Some hotwords might help. "
            f"The hotwords are {hotwords_str}. ASSISTANT: {transcription}"
        )

    return prompt
