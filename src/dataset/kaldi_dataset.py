import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.dataset_utils import decode_audio, resample_audio

logger = logging.getLogger(__name__)


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

class CSJCorpusDataset(torch.utils.data.Dataset):
    """CSJ (ESPnet ``dump/raw`` layout) as a dataset for the ASR pipeline.

    Reads Kaldi-style ``text`` / ``wav.scp`` / ``utt2spk`` / ``utt2num_samples``
    and yields the **same dict interface** as ``LibriSpeechDataset`` so it drops
    into ``build_vocab`` / ``WhisperCTCDataset`` unchanged:
    ``{"audio", "sampling_rate", "transcription", "speaker_id", "utterance_id"}``.

    Args:
        split: Sub-directory under *base_dir* (e.g.
            ``train_nodup_whisper_like_clean``, ``train_dev_whisper_like_clean``).
        base_dir: Root that ``wav.scp`` relative paths resolve against (the
            ``dump/raw`` dir, or the ``data/csj`` symlink to it).
        max_samples: Drop utterances longer than this many samples. Defaults to
            30 s (480000 @ 16 kHz) because Whisper truncates to a 30 s window, so
            longer clips would have CTC targets that cannot align. Set to 0/None
            to keep everything.
    """

    def __init__(
        self,
        split: str = "train_dev",
        base_dir: str = "/home/minh/espnet/egs2/csj/asr1/dump/raw/",
        max_samples: int = 480000,
    ):
        self.split = split
        self.base_dir = base_dir
        self.root_dir = os.path.join(self.base_dir, split)
        self.max_samples = max_samples
        self.utt2wav = {}
        self.utt2text = {}
        self.utt2spk = {}
        self.utt2num_samples = {}
        self._load_metadata()
        self._build_index()

    def _load_metadata(self):
        logger.info("Loading CSJ metadata | split=%s | root_dir=%s", self.split, self.root_dir)

        # 1. Load text
        text_path = os.path.join(self.root_dir, 'text')
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                for line in f:
                    utt, *txt = line.strip().split(maxsplit=1)
                    self.utt2text[utt] = txt[0] if txt else ""
        else:
            logger.warning("Missing text file: %s", text_path)

        # 2. Load utt2spk
        utt2spk_path = os.path.join(self.root_dir, 'utt2spk')
        if os.path.exists(utt2spk_path):
            with open(utt2spk_path, 'r', encoding='utf-8') as f:
                for line in f:
                    utt, spk = line.strip().split()
                    self.utt2spk[utt] = spk
        else:
            logger.warning("Missing utt2spk file: %s", utt2spk_path)

        
        # 3. Load wav.scp from all format.* folders
        wavscp_path = os.path.join(self.root_dir, 'wav.scp')
        if os.path.exists(wavscp_path):
            # wav.scp holds paths relative to the espnet recipe root, while base_dir points at dump/raw
            prefix = "dump/raw/"
            with open(wavscp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    utt, wav_path = line.strip().split(maxsplit=1)
                    if os.path.isabs(wav_path):
                        self.utt2wav[utt] = wav_path
                        continue
                    if wav_path.startswith(prefix):
                        wav_path = wav_path[len(prefix):]
                    self.utt2wav[utt] = os.path.join(self.base_dir, wav_path)
        else:
            logger.warning("Missing wav.scp file: %s", wavscp_path)

        # 4. Load utt2num_samples if exists
        utt2num_samples_path = os.path.join(self.root_dir, 'utt2num_samples')
        if os.path.exists(utt2num_samples_path):
            with open(utt2num_samples_path, 'r', encoding='utf-8') as f:
                for line in f:
                    utt, num = line.strip().split()
                    self.utt2num_samples[utt] = int(num)
        else:
            logger.warning("Missing utt2num_samples file: %s", utt2num_samples_path)

        logger.info(
            "Loaded CSJ metadata | split=%s | wavs=%d | texts=%d | spks=%d | num_samples=%d",
            self.split,
            len(self.utt2wav),
            len(self.utt2text),
            len(self.utt2spk),
            len(self.utt2num_samples),
        )

    def _build_index(self):
        """Keep only utterances with both audio and a non-empty transcript,
        dropping over-long clips. Cached so ``__getitem__`` is O(1)."""
        keys = []
        dropped_len = dropped_txt = 0
        for utt in self.utt2wav:
            if not self.utt2text.get(utt, "").strip():
                dropped_txt += 1
                continue
            if self.max_samples:
                n = self.utt2num_samples.get(utt)
                if n is not None and n > self.max_samples:
                    dropped_len += 1
                    continue
            keys.append(utt)
        self._utt_keys = keys
        logger.info(
            "CSJ index | split=%s | usable=%d (dropped: no-text=%d, >max_samples=%d)",
            self.split,
            len(self._utt_keys),
            dropped_txt,
            dropped_len,
        )

    def __len__(self):
        return len(self._utt_keys)

    def __getitem__(self, idx):
        utt = self._utt_keys[idx]
        wav_path = self.utt2wav[utt]
        try:
            # soundfile-first decode (no torchcodec/FFmpeg dependency); the CSJ
            # dump is already 16 kHz mono, so resample is usually a no-op.
            audio, sr = decode_audio(wav_path)
            audio = resample_audio(audio, sr, 16000)
        except Exception:
            logger.exception("Failed to load waveform | utt=%s | wav_path=%s", utt, wav_path)
            raise
        return {
            "audio": audio,
            "sampling_rate": 16000,
            "transcription": self.utt2text.get(utt, ""),
            "speaker_id": self.utt2spk.get(utt, ""),
            "utterance_id": utt,
        }