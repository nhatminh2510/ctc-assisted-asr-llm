import os
import torchaudio
from torch.utils.data import Dataset


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
