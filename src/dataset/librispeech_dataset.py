import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.dataset_utils import load_audio
from src.utils.train_utils import generate_training_hotwords
from src.dataset.asr_dataset import ASRDataset


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
