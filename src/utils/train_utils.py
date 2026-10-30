import sys
import torch
import torchaudio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import random
from pathlib import Path
import yaml
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.create_prompt import create_prompt_with_hotwords, detect_prompt_format


logger = logging.getLogger(__name__)


def _split_words(transcription: str, language: str) -> Tuple[List[str], str]:
    """Split a transcription into words and return ``(words, join_str)``.

    English splits on whitespace and re-joins with a space; Japanese has no
    spaces, so it is segmented with a morphological analyzer and re-joined with
    the empty string.
    """
    if language == "ja":
        # Imported lazily so English training does not depend on the JA stack.
        from src.utils.japanese import JapaneseTokenizer

        global _JA_TOKENIZER
        try:
            tok = _JA_TOKENIZER
        except NameError:
            tok = _JA_TOKENIZER = JapaneseTokenizer()
        return [t.surface for t in tok.segment(transcription)], ""
    return transcription.split(), " "


def generate_training_hotwords(
    transcription: str,
    p_keep: float = 0.5,
    n_phrases: int = 1,
    n_order_max: int = 4,
    language: str = "en",
) -> List[str]:
    """
    Tạo danh sách hotwords cho training từ transcription.

    Args:
        transcription: Câu phiên âm ground truth
        p_keep: Xác suất giữ lại câu để tạo hotwords
        n_phrases: Số cụm từ (phrases) được chọn từ mỗi câu
        n_order_max: Độ dài tối đa của n-gram
        language: ``"en"`` (whitespace split) or ``"ja"`` (morphological
            segmentation, joined without spaces).

    Returns:
        hotwords: Danh sách hotwords
    """
    hotwords = []

    # Với xác suất p_keep, lấy hotwords từ transcription
    if random.random() < p_keep:
        words, join_str = _split_words(transcription, language)

        if len(words) > 0:
            k = random.randint(1, n_phrases)
            for _ in range(k):
                # Chọn ngẫu nhiên độ dài n-gram
                n_order = random.randint(1, min(n_order_max, len(words)))

                # Chọn ngẫu nhiên vị trí bắt đầu
                if len(words) >= n_order:
                    start_idx = random.randint(0, len(words) - n_order)
                    phrase = join_str.join(words[start_idx : start_idx + n_order])
                    hotwords.append(phrase)

    return hotwords


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    loss: float,
    wer: float,
    save_path: str,
    is_best: bool = False,
    metric: str = "wer",
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
        metric: "wer" hoặc "cer" – metric của giá trị ``wer`` (key ``"wer"``
            được giữ để tương thích ngược)
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "wer": wer,
        "metric": metric,
        "loss": loss,
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")

    if is_best:
        best_path = str(Path(save_path).parent / "best_model.pt")
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved to {best_path}")


def _remap_legacy_encoder_keys(model, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate pre-``SpeechEncoder`` checkpoints.

    Older checkpoints stored the encoder directly under ``encoder.*``; the
    encoder is now wrapped by :class:`~src.models.encoder.SpeechEncoder`, so its
    weights live under ``encoder.model.*``. Re-prefix legacy keys so they still
    load. Projector / LLM keys are unchanged.
    """
    model_keys = set(model.state_dict().keys())
    wants_wrapped = any(k.startswith("encoder.model.") for k in model_keys)
    has_legacy = any(
        k.startswith("encoder.") and not k.startswith("encoder.model.")
        for k in state_dict
    )
    if not (wants_wrapped and has_legacy):
        return state_dict

    remapped: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.") and not k.startswith("encoder.model."):
            remapped["encoder.model." + k[len("encoder.") :]] = v
        else:
            remapped[k] = v
    logger.info("Remapped legacy 'encoder.*' checkpoint keys to 'encoder.model.*'")
    return remapped


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

    state_dict = _remap_legacy_encoder_keys(model, checkpoint["model_state_dict"])
    # strict=False: the frozen encoder is (re)loaded from its pretrained id, so
    # small key differences there are expected; the trained projector must load.
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, "missing_keys", [])
    unexpected = getattr(load_result, "unexpected_keys", [])
    if missing:
        logger.warning(
            "load_state_dict: %d missing key(s), e.g. %s", len(missing), missing[:5]
        )
    if unexpected:
        logger.warning(
            "load_state_dict: %d unexpected key(s), e.g. %s",
            len(unexpected),
            unexpected[:5],
        )
    proj_missing = [k for k in missing if k.startswith("projector.")]
    if proj_missing:
        logger.error("Projector weights missing from checkpoint: %s", proj_missing[:5])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float("inf"))
    wer = checkpoint.get("wer", float("inf"))
    metric = checkpoint.get("metric", "wer").upper()

    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(
        f"Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}, {metric}: {wer:.4f}"
    )

    return epoch, step, loss, wer


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


def get_collate_fn(
    encoder,
    decoder_processor,
    mode: str = "train",
    device: str = "cuda",
    p_keep: float = 0.5,
    n_order_max: int = 4,
    n_phrases: int = 1,
    language: str = "en",
):
    """Create collate function with per-batch shared hotword generation (matching reference).

    ``encoder`` is a :class:`~src.models.encoder.SpeechEncoder`; audio is
    processed through it so the batch is encoder-agnostic (WavLM waveform or
    Whisper log-mel are both exposed under the ``input_values`` key).

    ``language`` selects whitespace (``"en"``) vs morphological (``"ja"``)
    hotword segmentation and the prompt instruction language. The chat scaffold
    (Vicuna or Llama-3) is picked from ``decoder_processor``.
    """
    prompt_format = detect_prompt_format(decoder_processor)

    def _generate_batch_hotwords(batch):
        """Generate shared hotwords from all batch transcriptions (per-batch).

        Each sample contributes one n-gram with probability p_keep.
        All n-grams are collected into a single shared hotword list for the whole batch.
        """
        selected_ngrams_list = []
        for sample in batch:
            ngrams = generate_training_hotwords(
                sample["transcription"],
                p_keep=p_keep,
                n_phrases=n_phrases,
                n_order_max=n_order_max,
                language=language,
            )
            selected_ngrams_list.extend(ngrams)
        return selected_ngrams_list

    def _build_batch(batch, to_device=None):
        # 1. Process audio (encoder-agnostic: waveform for WavLM, mel for Whisper)
        encoder_inputs = encoder.process_audio(
            [sample["audio"] for sample in batch],
            sampling_rate=16000,
            device=to_device,
        )

        # 2. Generate per-batch shared hotwords (matching reference collator)
        batch_hotwords = _generate_batch_hotwords(batch)

        # 3. Create shared prompt (same for all samples in batch)
        shared_prompt_only = create_prompt_with_hotwords(
            batch_hotwords, language=language, prompt_format=prompt_format
        )

        # 4. Compute prompt_length (same for all samples since prompt is shared)
        prompt_ids = decoder_processor.encode(shared_prompt_only)
        prompt_length = len(prompt_ids)
        prompt_lengths = [prompt_length] * len(batch)

        # 5. Create full text: shared_prompt + individual transcription + EOS
        full_texts = [
            create_prompt_with_hotwords(
                batch_hotwords,
                sample["transcription"],
                language=language,
                prompt_format=prompt_format,
            )
            for sample in batch
        ]
        full_decoder_inputs = decoder_processor(
            full_texts,
            return_tensors="pt",
            padding=True,
        )
        if to_device:
            full_decoder_inputs = full_decoder_inputs.to(to_device)

        if mode == "train":
            return {
                "input_values": encoder_inputs["input_values"],
                "input_ids": full_decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs["encoder_attention_mask"],
                "decoder_attention_mask": full_decoder_inputs.attention_mask,
                "prompt_lengths": prompt_lengths,
            }
        elif mode == "eval":
            # Prompt-only for generation (same prompt for all samples)
            prompt_only_inputs = decoder_processor(
                [shared_prompt_only] * len(batch),
                return_tensors="pt",
                padding=True,
            )
            if to_device:
                prompt_only_inputs = prompt_only_inputs.to(to_device)

            return {
                "input_values": encoder_inputs["input_values"],
                # Full prompt+transcription for loss computation
                "input_ids": full_decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs["encoder_attention_mask"],
                "decoder_attention_mask": full_decoder_inputs.attention_mask,
                # Prompt-only for generation
                "generate_input_ids": prompt_only_inputs.input_ids,
                "generate_attention_mask": prompt_only_inputs.attention_mask,
                "prompt_lengths": prompt_lengths,
                "transcription": [sample["transcription"] for sample in batch],
            }
        else:
            raise ValueError(
                f"Invalid collate mode: {mode}. Expected 'train' or 'eval'."
            )

    def collate_fn_cpu(batch):
        return _build_batch(batch, to_device=None)

    def collate_fn_gpu(batch):
        return _build_batch(batch, to_device="cuda")

    if device == "cuda":
        return collate_fn_gpu
    else:
        return collate_fn_cpu


def get_linear_schedule(
    optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def tensorboard_log_dir(output_dir: str) -> str:
    from datetime import datetime
    import socket
    import os

    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()

    log_dir = os.path.join(output_dir, "runs", f"{timestamp}_{hostname}")
    return log_dir
