import torch
import torchaudio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import random
from pathlib import Path
import yaml


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def generate_training_hotwords(
    transcription: str, p_keep: float = 0.5, n_phrases: int = 1, n_order_max: int = 4
) -> List[str]:
    """
    Tạo danh sách hotwords cho training từ transcription.

    Args:
        transcription: Câu phiên âm ground truth
        p_keep: Xác suất giữ lại câu để tạo hotwords
        n_phrases: Số cụm từ (phrases) được chọn từ mỗi câu
        n_order_max: Độ dài tối đa của n-gram

    Returns:
        hotwords: Danh sách hotwords
    """
    hotwords = []

    # Với xác suất p_keep, lấy hotwords từ transcription
    if random.random() < p_keep:
        words = transcription.lower().split()

        if len(words) > 0:
            k = random.randint(1, n_phrases)
            for _ in range(k):
                # Chọn ngẫu nhiên độ dài n-gram
                n_order = random.randint(1, min(n_order_max, len(words)))

                # Chọn ngẫu nhiên vị trí bắt đầu
                if len(words) >= n_order:
                    start_idx = random.randint(0, len(words) - n_order)
                    phrase = " ".join(words[start_idx : start_idx + n_order])
                    hotwords.append(phrase)

    return hotwords


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
    is_best: bool = False,
) -> None:
    """
    Lưu checkpoint của model.

    Args:
        model: Model cần lưu
        optimizer: Optimizer
        epoch: Số epoch hiện tại
        step: Số training step hiện tại
        loss: Loss hiện tại
        save_path: Đường dẫn lưu checkpoint
        is_best: Có phải là best checkpoint không
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")

    if is_best:
        best_path = str(Path(save_path).parent / "best_model.pt")
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved to {best_path}")


def load_checkpoint(
    model, checkpoint_path: str, optimizer: Optional[Any] = None, device: str = "cuda"
) -> Tuple[int, int, float]:
    """
    Load checkpoint của model.

    Args:
        model: Model để load weights
        checkpoint_path: Đường dẫn đến checkpoint
        optimizer: Optimizer (optional)
        device: Device để load model

    Returns:
        epoch: Epoch của checkpoint
        step: Step của checkpoint
        loss: Loss của checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float("inf"))

    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}")

    return epoch, step, loss


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Tính Word Error Rate (WER).

    Args:
        predictions: Danh sách các phiên âm dự đoán
        references: Danh sách các phiên âm ground truth

    Returns:
        wer: Word Error Rate (tỷ lệ %)
    """
    try:
        import jiwer

        wer = jiwer.wer(references, predictions)
        return wer
    except ImportError:
        raise ImportError(
            "jiwer is required for WER computation. Install with: pip install jiwer"
        )


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Thiết lập logging.

    Args:
        log_file: Đường dẫn đến file log (optional)
        level: Mức độ logging
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        config: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Lưu configuration ra file YAML.

    Args:
        config: Dictionary chứa configuration
        config_path: Đường dẫn để lưu file
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to {config_path}")


def save_info(info: Dict[str, Any], info_path: str) -> None:
    """
    Lưu thông tin ra file YAML.

    Args:
        info: Dictionary chứa thông tin
        info_path: Đường dẫn để lưu file
    """
    with open(info_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Info saved to {info_path}")
