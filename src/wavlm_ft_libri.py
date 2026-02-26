"""WavLM fine-tuning script for LibriSpeech dataset.

This module implements fine-tuning of WavLM model for CTC-based speech recognition
on LibriSpeech dataset with custom learning rate scheduling.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from jiwer import wer as compute_wer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, Dataset, Subset
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    WavLMForCTC,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataset import LibriSpeechDataset

# ============================================================================
# Constants
# ============================================================================

# Training schedule
DEFAULT_MAX_STEPS = 80000
DEFAULT_WARMUP_STEPS = 8000
DEFAULT_CONSTANT_STEPS = 32000
DEFAULT_DECAY_STEPS = 40000
UNFREEZE_STEP = 10000

# Hyperparameters
DEFAULT_PEAK_LR = 3e-5
DEFAULT_BATCH_SIZE = 8
DEFAULT_ACCUMULATION_STEPS = 4
DEFAULT_WEIGHT_DECAY = 0.005
DECAY_FINAL_RATIO = 0.05

# Data processing
DEFAULT_SAMPLING_RATE = 16000
VOCAB_SAMPLING_SIZE = 10000
FEATURE_SIZE = 1

# Setup
DEFAULT_DATA_DIR = "/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech"
DEFAULT_OUTPUT_DIR = "exp/wavlm-ft-libri"
DEFAULT_MODEL = "microsoft/wavlm-large"

# Logger
logger: Optional[logging.Logger] = None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune WavLM model on LibriSpeech with CTC loss"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to LibriSpeech dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for model checkpoints and logs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=DEFAULT_PEAK_LR,
        help="Peak learning rate for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=DEFAULT_ACCUMULATION_STEPS,
        help="Number of steps to accumulate gradients before updating",
    )

    return parser.parse_args()


def _setup_logger(output_dir: str) -> logging.Logger:
    """Initialize and configure logger for training.

    Args:
        output_dir: Directory to save logs.

    Returns:
        logging.Logger: Configured logger instance.
    """
    global logger
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/logs/training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def _patched_load_rng(self, checkpoint: str) -> None:
    """Patched RNG state loader for checkpoint resumption.

    Args:
        checkpoint: Path to checkpoint directory.
    """
    rng_file = os.path.join(checkpoint, "rng_state.pth")
    if not os.path.isfile(rng_file):
        return

    try:
        checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        if torch.cuda.is_available():
            if (
                hasattr(self.args, "parallel_mode")
                and str(self.args.parallel_mode) == "distributed"
            ):
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
    except Exception as e:
        logger.warning(f"Could not load RNG state from checkpoint: {e}")


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for CTC training with dynamic padding.

    Handles padding of audio features and text labels for batch processing
    in CTC-based ASR models. Uses -100 token ID for loss masking.

    Attributes:
        processor: Wav2Vec2Processor for feature extraction and tokenization.
        padding: Whether to pad sequences ('longest' or True for max in batch).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Collate batch of features for model input.

        Args:
            features: List of preprocessed samples with 'input_values' and 'labels'.

        Returns:
            Dict with padded 'input_values' and 'labels' with -100 masking.
        """
        # Separate audio features and text labels
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad audio to longest in batch
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels to longest in batch
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Mask padding tokens (-100 ignores them in loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch


class UnfreezeCallback(TrainerCallback):
    """Callback to unfreeze model parameters at specified step."""

    def __init__(self):
        self.unfrozen = False

    def on_train_begin(self, args, state, control, model=None, **kwargs) -> None:
        """Check if should unfreeze at training start (for resume)."""
        if state.global_step >= UNFREEZE_STEP and not self.unfrozen:
            logger.info(
                f"Step {state.global_step} >= {UNFREEZE_STEP}: Unfreezing model parameters..."
            )
            self._unfreeze_model(model)
            self.unfrozen = True

    def on_step_end(self, args, state, control, model=None, **kwargs) -> None:
        """Called at end of each training step."""
        if state.global_step == UNFREEZE_STEP and not self.unfrozen:
            logger.info(f"Step {UNFREEZE_STEP}: Unfreezing model parameters...")
            self._unfreeze_model(model)
            self.unfrozen = True

    def _unfreeze_model(self, model) -> None:
        """Unfreeze all parameters except feature encoder."""
        for param in model.parameters():
            param.requires_grad = True
        model.freeze_feature_encoder()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")


def remove_special_characters(text: str) -> str:
    """Remove special characters from text for vocabulary.

    Args:
        text: Input text string.

    Returns:
        Cleaned text with special characters removed and lowercased.
    """
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
    return re.sub(chars_to_ignore_regex, "", text).lower()


def extract_vocab_from_dataset(dataset: Dataset) -> List[str]:
    """Extract unique characters from dataset transcriptions.

    Args:
        dataset: Dataset with 'transcription' field.

    Returns:
        Sorted list of unique characters in dataset.
    """
    all_chars = set()
    n_samples = len(dataset)
    logger.info(f"Extracting vocabulary from {n_samples} samples...")

    for i in tqdm(range(n_samples), desc="Building vocab"):
        sample = dataset[i]
        text = sample.get("transcription", "")
        cleaned_text = remove_special_characters(str(text))
        all_chars.update(cleaned_text)

    return sorted(list(all_chars))


def create_datasets(data_dir: str) -> tuple[Dataset, Dataset]:
    """Load LibriSpeech dataset splits.

    Args:
        data_dir: Path to LibriSpeech dataset root.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    logger.info("Loading LibriSpeech datasets...")

    train_splits = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]
    logger.info(f"  Train splits: {', '.join(train_splits)}")
    train_dataset = ConcatDataset(
        [LibriSpeechDataset(data_path=data_dir, split=s) for s in train_splits]
    )

    val_splits = ["dev-clean", "dev-other"]
    logger.info(f"  Val splits: {', '.join(val_splits)}")
    val_dataset = ConcatDataset(
        [LibriSpeechDataset(data_path=data_dir, split=s) for s in val_splits]
    )

    return train_dataset, val_dataset


def create_vocab(dataset: Dataset, output_dir: str) -> str:
    """Create CTC vocabulary from dataset transcriptions.

    Args:
        dataset: Dataset with transcriptions.
        output_dir: Directory to save vocab.json.

    Returns:
        Path to saved vocab.json file.
    """
    os.makedirs(output_dir, exist_ok=True)
    vocab_file = os.path.join(output_dir, "vocab.json")

    if os.path.exists(vocab_file):
        logger.info(f"Using existing vocab: {vocab_file}")
        return vocab_file

    # Sample subset for efficiency (LibriSpeech vocab is consistent)
    max_samples = min(len(dataset), VOCAB_SAMPLING_SIZE)
    if len(dataset) > max_samples:
        logger.info(
            f"Sampling {max_samples}/{len(dataset)} samples for vocab extraction"
        )
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        subset = Subset(dataset, indices)
    else:
        subset = dataset

    # Extract unique characters
    vocab_list = extract_vocab_from_dataset(subset)
    vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}

    # Add special tokens
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    for token in special_tokens:
        if token not in vocab_dict:
            vocab_dict[token] = len(vocab_dict)

    # Replace space with pipe (word delimiter)
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    elif "|" not in vocab_dict:
        vocab_dict["|"] = len(vocab_dict)

    logger.info(f"Vocabulary size: {len(vocab_dict)} tokens")

    # Save to JSON
    with open(vocab_file, "w") as f:
        json.dump(vocab_dict, f, indent=2)

    logger.info(f"Saved vocab to {vocab_file}")
    return vocab_file


class LazyPreprocessedDataset(Dataset):
    """Dataset wrapper that applies preprocessing on-the-fly.

    More memory efficient than preprocessing entire dataset upfront.

    Attributes:
        base_dataset: Underlying dataset.
        processor: Wav2Vec2Processor for feature extraction and tokenization.
    """

    def __init__(self, base_dataset: Dataset, processor: Wav2Vec2Processor) -> None:
        """Initialize lazy preprocessing dataset.

        Args:
            base_dataset: Dataset to wrap.
            processor: Processor for audio and text processing.
        """
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get preprocessed sample by index.

        Args:
            idx: Sample index.

        Returns:
            Preprocessed sample dict with 'input_values' and 'labels'.
        """
        sample = self.base_dataset[idx]

        # Process audio waveform
        audio = sample["audio"].squeeze(0).numpy()
        inputs = self.processor(audio, sampling_rate=sample["sampling_rate"])

        # Process text transcription
        text = remove_special_characters(sample["transcription"])
        with self.processor.as_target_processor():
            labels = self.processor(text)

        return {
            "input_values": inputs.input_values[0],  # 1D waveform
            "labels": labels.input_ids,  # Token IDs
        }


def compute_metrics(processor: Wav2Vec2Processor) -> Callable:
    """Create metrics computation function for evaluation.

    Args:
        processor: Wav2Vec2Processor for decoding predictions.

    Returns:
        Callable that computes WER from predictions.
    """

    def _compute_metrics(pred) -> Dict[str, float]:
        """Compute WER metric from model predictions.

        Args:
            pred: EvalPrediction with predictions and label_ids.

        Returns:
            Dict with 'wer' key.
        """
        # Convert logits to token IDs
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        logger.debug(
            f"Predictions shape: {pred_ids.shape}, "
            f"vocab size: {len(processor.tokenizer)}, "
            f"pred ID range: [{pred_ids.min()}, {pred_ids.max()}]"
        )

        if pred_ids.max() >= len(processor.tokenizer):
            logger.warning(
                f"Prediction ID {pred_ids.max()} exceeds vocab size "
                f"{len(processor.tokenizer)}"
            )

        # Replace loss masking tokens with pad token for decoding
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode to text
        try:
            pred_str = processor.batch_decode(pred_ids)
            label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            return {"wer": float("inf")}

        # Compute word error rate
        wer = compute_wer(pred_str, label_str)
        logger.debug(f"WER: {wer:.4f}")
        return {"wer": wer}

    return _compute_metrics


def get_lr_scheduler(
    optimizer,
    num_warmup_steps: int = DEFAULT_WARMUP_STEPS,
    num_constant_steps: int = DEFAULT_CONSTANT_STEPS,
    num_decay_steps: int = DEFAULT_DECAY_STEPS,
) -> LambdaLR:
    """Create custom learning rate scheduler with three phases.

    Schedule:
        1. Linear warmup: 0 to num_warmup_steps
        2. Constant LR: num_warmup_steps to num_warmup_steps + num_constant_steps
        3. Exponential decay: Remaining steps decay to DECAY_FINAL_RATIO

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Warmup phase duration.
        num_constant_steps: Constant phase duration.
        num_decay_steps: Decay phase duration.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        """Compute learning rate multiplier for given step.

        Args:
            current_step: Current training step.

        Returns:
            LR multiplier (between DECAY_FINAL_RATIO and 1.0).
        """
        # Phase 1: Linear warmup from 0 to 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Phase 2: Maintain constant LR
        elif current_step < num_warmup_steps + num_constant_steps:
            return 1.0

        # Phase 3: Exponential decay to DECAY_FINAL_RATIO
        else:
            decay_step = current_step - (num_warmup_steps + num_constant_steps)
            progress = float(decay_step) / float(num_decay_steps)
            decay_factor = DECAY_FINAL_RATIO**progress
            return max(DECAY_FINAL_RATIO, decay_factor)

    return LambdaLR(optimizer, lr_lambda)


def train_ssl(
    data_dir: str,
    output_dir: str,
    model_name: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    peak_lr: float = DEFAULT_PEAK_LR,
    constant_steps: int = DEFAULT_CONSTANT_STEPS,
    decay_steps: int = DEFAULT_DECAY_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    accumulation_steps: int = DEFAULT_ACCUMULATION_STEPS,
    checkpoint: Optional[str] = None,
) -> None:
    """Fine-tune WavLM model on LibriSpeech with custom learning schedule.

    Args:
        data_dir: Path to LibriSpeech dataset.
        output_dir: Path to save checkpoints and logs.
        model_name: HuggingFace model identifier.
        max_steps: Total training steps.
        warmup_steps: Number of warmup steps.
        peak_lr: Peak learning rate.
        constant_steps: Steps at constant LR.
        decay_steps: Steps for exponential decay.
        checkpoint: Path to resume from (optional).

    Raises:
        FileNotFoundError: If data directory doesn't exist.
    """
    # Initialize logger
    _setup_logger(output_dir)
    logger.info("-" * 70)
    logger.info("WavLM Fine-tuning Configuration")
    logger.info("-" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(
        f"Total steps: {max_steps} (warmup: {warmup_steps}, "
        f"constant: {constant_steps}, decay: {decay_steps})"
    )
    logger.info("-" * 70)

    # Load datasets
    logger.info("[1/8] Loading datasets...")
    train_dataset, val_dataset = create_datasets(data_dir)
    logger.info(f"  Train: {len(train_dataset):,} samples")
    logger.info(f"  Validation: {len(val_dataset):,} samples")

    # Create vocabulary
    logger.info("[2/8] Creating vocabulary...")
    vocab_file = create_vocab(train_dataset, output_dir)

    # Initialize tokenizer and processor
    logger.info("[3/8] Initializing processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=FEATURE_SIZE,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # Apply lazy preprocessing
    logger.info("[4/8] Preparing datasets...")
    train_dataset = LazyPreprocessedDataset(train_dataset, processor)
    val_dataset = LazyPreprocessedDataset(val_dataset, processor)

    # Initialize data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize model
    logger.info("[5/8] Initializing model...")
    model = WavLMForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
    )
    model = model.to("cuda")

    # Freeze feature encoder and all params initially
    model.freeze_feature_encoder()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Validate model config
    assert model.lm_head.out_features == len(processor.tokenizer), (
        f"Vocab mismatch: lm_head output {model.lm_head.out_features} != "
        f"tokenizer size {len(processor.tokenizer)}"
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded with {trainable:,} trainable / {total:,} total params")
    logger.info(f"  (Only lm_head trainable until step {UNFREEZE_STEP})")

    # Training arguments
    logger.info("[6/8] Setting up optimizer and scheduler...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        # Schedule
        max_steps=max_steps,
        lr_scheduler_type="linear",  # Will be overridden
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        # Optimization
        learning_rate=peak_lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        # Efficiency
        fp16=False,  # WavLM can be unstable with mixed precision with fp16
        gradient_checkpointing=True,
        group_by_length=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=5000,
        logging_steps=100,
        save_steps=10000,
        save_strategy="steps",
        save_total_limit=2,
        # Performance
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)
    scheduler = get_lr_scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_constant_steps=constant_steps,
        num_decay_steps=decay_steps,
    )
    logger.info(f"  Optimizer: AdamW (LR: {peak_lr})")
    logger.info(
        f"  Schedule: Linear warmup ({warmup_steps}), "
        f"constant ({constant_steps}), decay to {DECAY_FINAL_RATIO} ({decay_steps})"
    )

    # Initialize trainer
    logger.info("[7/8] Initializing trainer...")
    Trainer._load_rng_state = _patched_load_rng
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.feature_extractor,
        compute_metrics=compute_metrics(processor),
        optimizers=(optimizer, scheduler),
        callbacks=[UnfreezeCallback()],
    )
    logger.info("  Trainer initialized")

    # Start training
    logger.info("[8/8] Starting training...")
    try:
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    logger.info(f"Saving final model to {final_dir}...")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("Training complete!")


if __name__ == "__main__":
    args = parse_arguments()
    start_time = time.time()

    try:
        train_ssl(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            peak_lr=args.learning_rate,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            checkpoint=args.checkpoint,
        )
    except Exception:
        if logger:
            logger.exception("Training failed")
        raise
    finally:
        if logger:
            elapsed = time.time() - start_time
            logger.info(
                f"Total time: {elapsed/3600:.2f} hours - {elapsed/(3600*24):.2f} days"
            )
