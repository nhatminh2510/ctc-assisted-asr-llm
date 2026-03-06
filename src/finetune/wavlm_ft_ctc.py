"""Config-driven WavLM CTC fine-tuning on LibriSpeech.

Reads all hyperparameters from a YAML config (configs/tuning/wavlm-ctc.yaml)
so the script itself stays short and declarative.

Reference: https://arxiv.org/pdf/2411.06437
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from jiwer import wer as compute_wer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, Dataset, Subset
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    WavLMForCTC,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset.librispeech_dataset import LibriSpeechDataset
from src.utils.config_utils import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)

# Characters stripped before tokenisation (same set as Kaldi / wav2vec2 convention)
_SPECIAL_CHARS_RE = re.compile(r"[\,\?\.\!\-\;\:\"]")


# ============================================================================
# Helpers
# ============================================================================


def clean_text(text: str) -> str:
    """Lower-case and strip punctuation."""
    return _SPECIAL_CHARS_RE.sub("", text).lower()


# ============================================================================
# Vocabulary
# ============================================================================


def build_vocab(dataset: Dataset, cfg: Dict[str, Any]) -> str:
    """Build character-level CTC vocabulary from dataset transcriptions.

    Saves ``vocab.json`` to *output_dir*. Returns the path.
    If the file already exists it is reused.
    """
    output_dir: str = cfg["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")

    if os.path.exists(vocab_path):
        logger.info(f"Reusing existing vocab: {vocab_path}")
        return vocab_path

    # Sub-sample for speed – LibriSpeech character set is stable
    max_n = min(len(dataset), cfg["data"].get("vocab_sample_size", 10000))
    indices = np.random.choice(len(dataset), max_n, replace=False)
    subset = Subset(dataset, indices)

    chars: set[str] = set()
    for i in tqdm(range(len(subset)), desc="Building vocab"):
        text = clean_text(str(subset[i].get("transcription", "")))
        chars.update(text)

    vocab = {c: i for i, c in enumerate(sorted(chars))}

    # Special tokens
    for tok in cfg["model"].get(
        "vocab_special_tokens", ["<pad>", "<unk>", "<s>", "</s>"]
    ):
        if tok not in vocab:
            vocab[tok] = len(vocab)

    # Word-delimiter: space → |
    if " " in vocab:
        vocab["|"] = vocab.pop(" ")
    elif "|" not in vocab:
        vocab["|"] = len(vocab)

    with open(vocab_path, "w") as fh:
        json.dump(vocab, fh, indent=2)
    logger.info(f"Vocab ({len(vocab)} tokens) saved to {vocab_path}")
    return vocab_path


# ============================================================================
# Processor factory
# ============================================================================


def create_processor(vocab_path: str, cfg: Dict[str, Any]) -> Wav2Vec2Processor:
    """Create tokenizer + feature extractor wrapped in a Wav2Vec2Processor."""
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token=cfg["model"].get("word_delimiter_token", "|"),
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=cfg["data"]["sampling_rate"],
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# ============================================================================
# Dataset wrapper
# ============================================================================


class CTCDataset(Dataset):
    """Lazy preprocessing wrapper: audio → features, text → token IDs."""

    def __init__(self, base: Dataset, processor: Wav2Vec2Processor) -> None:
        self.base = base
        self.processor = processor

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        audio = (
            sample["audio"]
            if isinstance(sample["audio"], np.ndarray)
            else sample["audio"].squeeze(0).numpy()
        )
        inputs = self.processor(audio, sampling_rate=sample["sampling_rate"])
        text = clean_text(sample["transcription"])
        labels = self.processor.tokenizer(text)
        return {"input_values": inputs.input_values[0], "labels": labels.input_ids}


# ============================================================================
# Data collator
# ============================================================================


@dataclass
class CTCCollator:
    """Pad audio features and labels for CTC training.

    Labels are padded with -100 so they are ignored in the CTC loss.
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_feats = [{"input_values": f["input_values"]} for f in features]
        label_feats = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_feats, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_feats, padding=self.padding, return_tensors="pt"
        )
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return batch


# ============================================================================
# LR schedule (warmup → constant → exponential decay)
# ============================================================================


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]
) -> LambdaLR:
    """Three-phase LR schedule driven by *cfg['schedule']*.

    1. Linear warmup  (0 → warmup_steps)
    2. Constant        (warmup_steps → warmup + constant)
    3. Exponential decay to ``decay_final_ratio`` of peak
    """
    sched = cfg["schedule"]
    warmup = sched["warmup_steps"]
    constant = sched["constant_steps"]
    decay = sched["decay_steps"]
    final = sched.get("decay_final_ratio", 0.05)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        if step < warmup + constant:
            return 1.0
        progress = (step - warmup - constant) / max(1, decay)
        return max(final, final**progress)

    return LambdaLR(optimizer, _lr_lambda)


# ============================================================================
# Unfreeze callback
# ============================================================================


class UnfreezeCallback(TrainerCallback):
    """Unfreeze transformer layers at a configurable step."""

    def __init__(self, unfreeze_step: int) -> None:
        self.unfreeze_step = unfreeze_step
        self._done = False

    # ---- hooks ----
    def on_train_begin(self, args, state, control, model=None, **kw):
        if state.global_step >= self.unfreeze_step:
            self._unfreeze(model, state.global_step)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step >= self.unfreeze_step and not self._done:
            self._unfreeze(model, state.global_step)

    # ---- logic ----
    def _unfreeze(self, model, step: int) -> None:
        if self._done:
            return
        logger.info(f"[step {step}] Unfreezing transformer layers")
        for p in model.parameters():
            p.requires_grad = True
        model.freeze_feature_encoder()  # keep CNN frozen
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"  Trainable: {trainable:,} / {total:,}")
        self._done = True


# ============================================================================
# Metrics
# ============================================================================


def make_compute_metrics(processor: Wav2Vec2Processor) -> Callable:
    """Return a function that computes WER from CTC logits."""

    def _fn(pred) -> Dict[str, float]:
        ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        try:
            hyps = processor.batch_decode(ids)
            refs = processor.batch_decode(pred.label_ids, group_tokens=False)
        except Exception as exc:
            logger.error(f"Decode error: {exc}")
            return {"wer": float("inf")}
        return {"wer": compute_wer(hyps, refs)}

    return _fn


# ============================================================================
# Main training routine
# ============================================================================


def train(cfg: Dict[str, Any], checkpoint: Optional[str] = None) -> None:
    """Run the full WavLM-CTC fine-tuning pipeline.

    Args:
        cfg: Merged YAML config dict.
        checkpoint: Optional path to resume from.
    """
    out_dir = cfg["output"]["output_dir"]
    os.makedirs(f"{out_dir}/logs", exist_ok=True)
    setup_logging(log_file=os.path.join(out_dir, "logs", "training.log"))

    set_seed(cfg.get("seed", 42))

    # ---- log config summary ----
    logger.info("=" * 70)
    logger.info("WavLM-CTC Fine-tuning  (config-driven)")
    logger.info("=" * 70)
    logger.info(f"  Model       : {cfg['model']['name']}")
    logger.info(f"  Data dir    : {cfg['data']['data_dir']}")
    logger.info(f"  Output dir  : {out_dir}")
    sched = cfg["schedule"]
    logger.info(
        f"  Steps       : {sched['max_steps']}  "
        f"(warm {sched['warmup_steps']} / const {sched['constant_steps']} / decay {sched['decay_steps']})"
    )
    logger.info(f"  Peak LR     : {cfg['optimizer']['peak_lr']}")
    logger.info(
        f"  Batch size  : {cfg['training']['batch_size']} x {cfg['training']['gradient_accumulation_steps']} accum"
    )
    logger.info("=" * 70)

    # ---- datasets ----
    data_dir = cfg["data"]["data_dir"]
    train_ds = ConcatDataset(
        [
            LibriSpeechDataset(data_path=data_dir, split=s)
            for s in cfg["data"]["train_splits"]
        ]
    )
    val_ds = ConcatDataset(
        [
            LibriSpeechDataset(data_path=data_dir, split=s)
            for s in cfg["data"]["val_splits"]
        ]
    )
    logger.info(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # ---- vocab & processor ----
    vocab_path = build_vocab(train_ds, cfg)
    processor = create_processor(vocab_path, cfg)

    # ---- wrap datasets ----
    train_ds = CTCDataset(train_ds, processor)
    val_ds = CTCDataset(val_ds, processor)

    # ---- model ----
    model = WavLMForCTC.from_pretrained(
        cfg["model"]["name"],
        ctc_loss_reduction=cfg["model"].get("ctc_loss_reduction", "mean"),
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    ).cuda()

    # Phase-1 freeze: only lm_head trainable
    model.freeze_feature_encoder()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True

    assert model.lm_head.out_features == len(
        processor.tokenizer
    ), f"lm_head output ({model.lm_head.out_features}) != tokenizer size ({len(processor.tokenizer)})"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model loaded — {trainable:,} trainable / {total:,} total  (lm_head only until step {cfg['freeze']['unfreeze_step']})"
    )

    # ---- training arguments ----
    ev = cfg["eval"]
    tr = cfg["training"]
    prec = cfg.get("precision", {})
    dl = cfg.get("dataloader", {})

    training_args = TrainingArguments(
        output_dir=out_dir,
        max_steps=sched["max_steps"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=cfg["optimizer"]["peak_lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        warmup_steps=sched["warmup_steps"],
        lr_scheduler_type="linear",  # overridden by custom scheduler
        fp16=prec.get("fp16", False),
        bf16=prec.get("bf16", False),
        gradient_checkpointing=tr.get("gradient_checkpointing", True),
        train_sampling_strategy=tr.get("train_sampling_strategy", "random"),
        eval_strategy="steps",
        eval_steps=ev["eval_steps"],
        logging_steps=ev["logging_steps"],
        save_steps=ev["save_steps"],
        save_strategy="steps",
        save_total_limit=ev.get("save_total_limit", 2),
        load_best_model_at_end=ev.get("load_best_model_at_end", True),
        metric_for_best_model=ev.get("metric_for_best_model", "wer"),
        greater_is_better=ev.get("greater_is_better", False),
        dataloader_num_workers=dl.get("num_workers", 4),
        dataloader_pin_memory=dl.get("pin_memory", True),
        report_to=cfg["output"].get("report_to", ["tensorboard"]),
        seed=cfg.get("seed", 42),
    )

    # ---- optimizer & scheduler ----
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["peak_lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=opt_cfg.get("eps", 1e-8),
    )
    scheduler = build_lr_scheduler(optimizer, cfg)

    # ---- trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=CTCCollator(processor=processor),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor.feature_extractor,
        compute_metrics=make_compute_metrics(processor),
        optimizers=(optimizer, scheduler),
        callbacks=[UnfreezeCallback(cfg["freeze"]["unfreeze_step"])],
    )

    # ---- train ----
    logger.info("Starting training …")
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except Exception:
        logger.exception("Training failed")
        raise

    # ---- save ----
    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WavLM CTC fine-tuning (config-driven)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )
    # Allow lightweight CLI overrides (flat key=value)
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="key=value overrides (e.g. optimizer.peak_lr=1e-4)",
    )
    return p.parse_args()


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> None:
    """Apply ``key.subkey=value`` overrides to *cfg* in-place."""
    for item in overrides:
        key, _, val = item.partition("=")
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Attempt numeric conversion
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[parts[-1]] = val


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()

    cfg = load_config(args.config)
    if args.override:
        _apply_overrides(cfg, args.override)

    try:
        train(cfg, checkpoint=args.checkpoint)
    except Exception:
        logging.getLogger(__name__).exception("Training failed")
        raise
    finally:
        elapsed = time.time() - t0
        logging.getLogger(__name__).info(f"Total time: {elapsed / 3600:.2f}h")
