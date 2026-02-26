import os
import sys
from pathlib import Path
import torchaudio
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import load_audio, generate_training_hotwords


class KaldiDataset(Dataset):
    def __init__(self, base_dir: str, split: str = "dev"):
        self.split = split
        self.base_dir = base_dir
        self.root_dir = os.path.join(self.base_dir, split)
        self.utt2wav = {}
        self.utt2text = {}
        self.utt2spk = {}
        self.utt2num_samples = {}
        self._load_metadata()
        self._utt_keys = list(self.utt2wav.keys())

    def _load_metadata(self):
        # 1. Load text
        text_path = os.path.join(self.root_dir, "text")
        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                for line in f:
                    utt, *txt = line.strip().split(maxsplit=1)
                    self.utt2text[utt] = txt[0] if txt else ""

        # 2. Load utt2spk
        utt2spk_path = os.path.join(self.root_dir, "utt2spk")
        if os.path.exists(utt2spk_path):
            with open(utt2spk_path, "r") as f:
                for line in f:
                    utt, spk = line.strip().split()
                    self.utt2spk[utt] = spk

        # 3. Load wav.scp from all format.* folders
        wavscp_path = os.path.join(self.root_dir, "wav.scp")
        if os.path.exists(wavscp_path):
            with open(wavscp_path, "r") as f:
                for line in f:
                    utt, wav_path = line.strip().split(maxsplit=1)
                    if self.split == "eval":
                        prefix = "dump/raw/eval/"
                        if wav_path.startswith(prefix):
                            wav_path = wav_path[len(prefix) :]
                        self.utt2wav[utt] = os.path.join(self.root_dir, wav_path)

                    elif self.split in ["train", "dev"]:
                        prefix = f"dump/raw/"
                        if wav_path.startswith(prefix):
                            wav_path = wav_path[len(prefix) :]
                        self.utt2wav[utt] = os.path.join(self.base_dir, wav_path)

        # 4. Load utt2num_samples if exists
        utt2num_samples_path = os.path.join(self.root_dir, "utt2num_samples")
        if os.path.exists(utt2num_samples_path):
            with open(utt2num_samples_path, "r") as f:
                for line in f:
                    utt, num = line.strip().split()
                    self.utt2num_samples[utt] = int(num)

    def __len__(self):
        return len(self.utt2wav)

    def __getitem__(self, idx):
        utt = self._utt_keys[idx]
        wav_path = self.utt2wav[utt]
        waveform, sr = torchaudio.load(wav_path)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sample = {
            "utt_id": utt,
            "text": self.utt2text.get(utt, ""),
            "spk": self.utt2spk.get(utt, ""),
            "num_samples": self.utt2num_samples.get(utt, None),
        }
        return (
            waveform,
            16000,
            sample["text"],
            sample["spk"],
            sample["num_samples"],
            sample["utt_id"],
        )


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


class LibriSpeechDataset(ASRDataset):
    """
    Dataset class for Librispeech data.

    Expected structure:
    data/Librispeech/
    ├── train-clean-100/
    ├── train-clean-360/
    ├── dev-clean/
    └── test-clean/
    """

    def __init__(
        self,
        data_path: str,
        config: Dict[str, Any] = {},
        split: str = "train-clean-100",
    ):
        """
        Initialize Librispeech dataset.

        Args:
            data_path: Path to Librispeech directory
            split: Data split ('train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean')
        """
        self.data_path = Path(data_path)
        self.split = split
        self.split_path = self.data_path / split

        if not self.split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.split_path}")

        # Load transcripts
        self.samples = []
        self._load_transcripts()
        super().__init__(data_list=self.samples, config=config)

    def _load_transcripts(self):
        """Load transcripts from Librispeech structure."""
        # Librispeech has structure: speaker_id/chapter_id/speaker_id-chapter_id-utterance_id
        for speaker_dir in self.split_path.iterdir():
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue

                chapter_id = chapter_dir.name

                # Find .trans.txt file containing transcripts
                trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                if not trans_file.exists():
                    continue

                # Parse transcripts
                with open(trans_file, "r") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            utterance_id, transcript = parts
                            audio_file = chapter_dir / f"{utterance_id}.flac"

                            if audio_file.exists():
                                self.samples.append(
                                    {
                                        "audio_path": str(audio_file),
                                        "transcription": transcript,
                                        "speaker_id": speaker_id,
                                        "utterance_id": utterance_id,
                                    }
                                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]

        # Load audio
        waveform = load_audio(sample["audio_path"], target_sr=self.sample_rate)
        transcription = sample["transcription"]
        audio = waveform.squeeze(0).numpy()
        # Tạo hotwords cho training
        hotwords = generate_training_hotwords(
            transcription,
            p_keep=self.p_keep,
            n_phrases=self.n_phrases,
            n_order_max=self.n_order_max,
        )

        return {
            "audio": audio,
            "sampling_rate": self.sample_rate,
            "transcription": transcription,
            "hotwords": hotwords,
            "speaker_id": sample["speaker_id"],
            "utterance_id": sample["utterance_id"],
        }
