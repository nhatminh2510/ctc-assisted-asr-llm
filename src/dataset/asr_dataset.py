import sys
from pathlib import Path
from typing import Dict, Any, List
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.dataset_utils import load_audio
from src.utils.train_utils import generate_training_hotwords


class ASRDataset(Dataset):
    """
    Dataset cho ASR training.

    Args:
        data_list: Danh sách các sample, mỗi sample là dict chứa:
                   {'audio_path': str, 'transcription': str}
        p_keep: Xác suất giữ lại câu để tạo hotwords
        n_phrases: Số cụm từ được chọn từ mỗi câu
        n_order_max: Độ dài tối đa của n-gram
    """

    def __init__(
        self,
        data_list: List[Dict[str, str]],
        config: Dict[str, Any] = {},
        p_keep: float = 0.5,
        n_phrases: int = 1,
        n_order_max: int = 4,
        sample_rate: int = 16000,
    ):
        self.data_list = data_list
        self.p_keep = config.get("p_keep", p_keep)
        self.n_phrases = config.get("n_phrases", n_phrases)
        self.n_order_max = config.get("n_order_max", n_order_max)
        self.sample_rate = config.get("sample_rate", sample_rate)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data_list[idx]

        # Load audio
        audio = load_audio(sample["audio_path"])
        transcription = sample["transcription"]

        # Tạo hotwords cho training
        hotwords = generate_training_hotwords(
            transcription,
            p_keep=self.p_keep,
            n_phrases=self.n_phrases,
            n_order_max=self.n_order_max,
        )

        return {"audio": audio, "transcription": transcription, "hotwords": hotwords}
