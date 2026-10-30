import numpy as np
import torch
import torchaudio
import logging
from typing import Any, Dict, Tuple


logger = logging.getLogger(__name__)


def decode_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Decode an audio file to ``(mono float32 [num_samples], sample_rate)``.

    Prefers ``soundfile`` (libsndfile: native WAV/FLAC, no FFmpeg). This avoids
    ``torchaudio.load``, which on recent torch (>=2.9, the pytorch 25.11 images)
    delegates to ``torchcodec`` and crashes when FFmpeg's shared libs are absent
    or torchcodec's ABI mismatches (``undefined symbol: torch_from_blob``).
    Falls back to ``torchaudio.load`` for formats/containers where soundfile is
    unavailable.
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:  # (num_samples, channels) -> mono
            audio = audio.mean(axis=1)
        return np.ascontiguousarray(audio, dtype=np.float32), sr
    except Exception as sf_exc:
        logger.debug("soundfile failed for %s (%s); trying torchaudio", audio_path, sf_exc)
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).numpy().astype(np.float32), sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample a mono float32 array. Uses ``torchaudio.functional`` (pure tensor
    math, no codec) so it works even when the torchaudio decode path is broken."""
    if orig_sr == target_sr:
        return audio
    resampled = torchaudio.functional.resample(
        torch.from_numpy(audio), orig_sr, target_sr
    )
    return resampled.numpy().astype(np.float32)


def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load file audio và resample về target sample rate.

    Args:
        audio_path: Đường dẫn đến file audio
        target_sr: Target sample rate (mặc định: 16000 Hz)

    Returns:
        audio_tensor: Tensor shape (1, num_samples)
    """
    try:
        audio, sample_rate = decode_audio(audio_path)
        audio = resample_audio(audio, sample_rate, target_sr)
        return torch.from_numpy(audio).unsqueeze(0)  # (1, num_samples)
    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {e}")
        raise


def to_cuda(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Đẩy tất cả Tensor trong batch lên device, giữ nguyên các kiểu dữ liệu khác."""
    return {
        k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
