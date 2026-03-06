import torch
import torchaudio
import logging


logger = logging.getLogger(__name__)


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
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample nếu cần
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )
            waveform = resampler(waveform)

        # Chuyển stereo thành mono nếu cần
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {e}")
        raise
