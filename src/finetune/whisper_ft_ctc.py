"""Config-driven Whisper-encoder CTC fine-tuning for Japanese kana output.

Whisper is natively a seq2seq model and its decoder transcribes into ordinary
Japanese orthography (kanji+kana). This script instead takes the **Whisper
encoder**, bolts a CTC head on top, and trains it with CTC loss against **kana**
targets — so the resulting model emits a pure-kana hypothesis directly. That
hypothesis is exactly what the reading-based hotword filter
(``src/inference/hotword_filter_ja.py``) consumes to pick contextual hotwords
for the LLM prompt.

There is no ``WhisperForCTC`` in transformers, so :class:`WhisperEncoderForCTC`
is defined here: ``WhisperModel.encoder`` -> dropout -> Linear(d_model, vocab) ->
``F.ctc_loss`` (blank = pad id), returning a HF-``Trainer``-compatible output.

Kana targets are produced with :func:`src.utils.japanese.to_katakana`, which
needs a reading backend (fugashi / SudachiPy / pykakasi) — run this inside a
container that ships one (e.g. ``prefix-tree.sif``).

Mirrors ``wavlm_ft_ctc.py`` (same config layout, LR schedule, unfreeze
callback) so the two are easy to compare.
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
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, Dataset, Subset
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    Wav2Vec2CTCTokenizer,
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers.modeling_outputs import CausalLMOutput

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset.kaldi_dataset import CSJCorpusDataset
from src.utils.config_utils import load_config, set_seed, setup_logging
from src.utils.japanese import to_katakana, has_reading_backend

logger = logging.getLogger(__name__)

# Latin punctuation stripped before kana conversion; Japanese punctuation is
# dropped by the analyzer anyway.
_LATIN_PUNCT_RE = re.compile(r"[\,\?\.\!\-\;\:\"'`]")
_KANA_CHAR_RE = re.compile(r"[぀-ヿー]")


# ============================================================================
# Text -> kana target
# ============================================================================
def _is_kana_char(ch: str) -> bool:
    return bool(_KANA_CHAR_RE.match(ch))


def text_to_kana(text: str) -> str:
    """Convert a transcription to a normalised **katakana** target string.

    Reading conversion is done by :func:`to_katakana` (analyzer-backed). We keep
    only kana characters so the CTC vocabulary stays tiny and consistent with
    the hotword filter, which also normalises readings to katakana.
    """
    text = _LATIN_PUNCT_RE.sub("", text)
    kana = to_katakana(text)  # katakana, whitespace-free
    return "".join(ch for ch in kana if _is_kana_char(ch))


# ============================================================================
# Vocabulary (kana, character-level)
# ============================================================================
def build_vocab(dataset: Dataset, cfg: Dict[str, Any]) -> str:
    """Build a kana CTC vocabulary from the (kana-converted) transcriptions."""
    output_dir: str = cfg["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")

    if os.path.exists(vocab_path):
        logger.info(f"Reusing existing vocab: {vocab_path}")
        return vocab_path

    max_n = min(len(dataset), cfg["data"].get("vocab_sample_size", 20000))
    indices = np.random.choice(len(dataset), max_n, replace=False)
    subset = Subset(dataset, indices)

    chars: set = set()
    for i in tqdm(range(len(subset)), desc="Building kana vocab"):
        chars.update(text_to_kana(str(subset[i].get("transcription", ""))))

    vocab = {c: i for i, c in enumerate(sorted(chars))}
    for tok in cfg["model"].get("vocab_special_tokens", ["<pad>", "<unk>"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    # Word-delimiter kept for tokenizer compatibility (kana has no spaces).
    if "|" not in vocab:
        vocab["|"] = len(vocab)

    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)
    logger.info(f"Kana vocab ({len(vocab)} tokens) saved to {vocab_path}")
    return vocab_path


# ============================================================================
# Custom model: Whisper encoder + CTC head
# ============================================================================
class WhisperEncoderForCTC(nn.Module):
    """Whisper encoder tower with a linear CTC head.

    Forward accepts ``input_features`` (log-mel, ``(B, n_mels, 3000)``), an
    optional mel-frame ``attention_mask`` (used only to compute CTC input
    lengths), and ``labels`` padded with ``-100``. Returns a
    :class:`~transformers.modeling_outputs.CausalLMOutput` so it drops straight
    into a HF :class:`~transformers.Trainer`.
    """

    def __init__(
        self,
        whisper_id: str,
        vocab_size: int,
        pad_token_id: int,
        ctc_loss_reduction: str = "mean",
        ctc_zero_infinity: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        base = WhisperModel.from_pretrained(whisper_id)
        self.encoder = base.get_encoder()
        del base  # free the unused decoder
        self.config = self.encoder.config
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(self.config.d_model, vocab_size)
        self.pad_token_id = pad_token_id
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.vocab_size = vocab_size

    # ---- freezing helpers (parallel to Wav2Vec2/WavLM) ----
    def freeze_feature_encoder(self) -> None:
        """Keep the two conv front-end layers frozen."""
        for p in self.encoder.conv1.parameters():
            p.requires_grad = False
        for p in self.encoder.conv2.parameters():
            p.requires_grad = False

    def _output_lengths(self, attention_mask, batch, frames, device):
        if attention_mask is None:
            return torch.full((batch,), frames, dtype=torch.long, device=device)
        # Whisper's 2nd conv has stride 2: out = floor((L - 1) / 2) + 1.
        mel_len = attention_mask.sum(-1)
        lengths = torch.div(mel_len - 1, 2, rounding_mode="floor") + 1
        return lengths.clamp(min=1, max=frames)

    def forward(self, input_features, attention_mask=None, labels=None, **kwargs):
        hidden = self.encoder(input_features).last_hidden_state
        hidden = self.dropout(hidden)
        logits = self.lm_head(hidden)  # (B, T, V)

        loss = None
        if labels is not None:
            B, T = logits.size(0), logits.size(1)
            input_lengths = self._output_lengths(attention_mask, B, T, logits.device)
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flat_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
            # cuDNN CTC is picky about dtypes/shapes; the native path is safe.
            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flat_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.pad_token_id,
                    reduction=self.ctc_loss_reduction,
                    zero_infinity=self.ctc_zero_infinity,
                )
        return CausalLMOutput(loss=loss, logits=logits)


# ============================================================================
# Dataset wrapper: audio -> log-mel, transcript -> kana ids
# ============================================================================
class WhisperCTCDataset(Dataset):
    def __init__(self, base, feature_extractor, tokenizer) -> None:
        self.base = base
        self.fe = feature_extractor
        self.tok = tokenizer

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        audio = sample["audio"]
        if not isinstance(audio, np.ndarray):
            audio = audio.squeeze(0).numpy()
        feats = self.fe(
            audio,
            sampling_rate=sample["sampling_rate"],
            return_tensors="np",
            return_attention_mask=True,
        )
        kana = text_to_kana(sample["transcription"])
        labels = self.tok(kana).input_ids
        return {
            "input_features": feats.input_features[0],
            "attention_mask": feats.attention_mask[0],
            "labels": labels,
        }


@dataclass
class WhisperCTCCollator:
    tokenizer: Wav2Vec2CTCTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Whisper features are fixed-length (n_mels, 3000) -> just stack.
        input_features = torch.tensor(
            np.stack([f["input_features"] for f in features]), dtype=torch.float
        )
        attention_mask = torch.tensor(
            np.stack([f["attention_mask"] for f in features]), dtype=torch.long
        )
        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(
            label_feats, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================================
# LR schedule (warmup -> constant -> exponential decay)
# ============================================================================
def build_lr_scheduler(optimizer, cfg: Dict[str, Any]) -> LambdaLR:
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
    """Unfreeze the Whisper encoder transformer at a configurable step."""

    def __init__(self, unfreeze_step: int) -> None:
        self.unfreeze_step = unfreeze_step
        self._done = False

    def on_train_begin(self, args, state, control, model=None, **kw):
        if state.global_step >= self.unfreeze_step:
            self._unfreeze(model, state.global_step)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step >= self.unfreeze_step and not self._done:
            self._unfreeze(model, state.global_step)

    def _unfreeze(self, model, step: int) -> None:
        if self._done:
            return
        logger.info(f"[step {step}] Unfreezing Whisper encoder layers")
        for p in model.parameters():
            p.requires_grad = True
        model.freeze_feature_encoder()  # keep conv front-end frozen
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"  Trainable: {trainable:,} / {total:,}")
        self._done = True


# ============================================================================
# Metrics (character error rate on kana)
# ============================================================================
def make_compute_metrics(tokenizer: Wav2Vec2CTCTokenizer) -> Callable:
    try:
        from jiwer import cer as _cer
    except Exception:  # pragma: no cover
        _cer = None

    def _fn(pred) -> Dict[str, float]:
        ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        try:
            hyps = tokenizer.batch_decode(ids)
            refs = tokenizer.batch_decode(label_ids, group_tokens=False)
        except Exception as exc:
            logger.error(f"Decode error: {exc}")
            return {"cer": float("inf")}
        if _cer is not None:
            refs = [r if r.strip() else "。" for r in refs]
            return {"cer": _cer(refs, hyps)}
        import editdistance

        tot_d = tot_n = 0
        for h, r in zip(hyps, refs):
            tot_d += editdistance.eval(list(h), list(r))
            tot_n += max(1, len(r))
        return {"cer": tot_d / tot_n}

    return _fn


# ============================================================================
# Main training routine
# ============================================================================
def train(cfg: Dict[str, Any], checkpoint: Optional[str] = None) -> None:
    out_dir = cfg["output"]["output_dir"]
    os.makedirs(f"{out_dir}/logs", exist_ok=True)
    setup_logging(log_file=os.path.join(out_dir, "logs", "training.log"))
    set_seed(cfg.get("seed", 42))

    if not has_reading_backend():
        logger.error(
            "No kana backend (fugashi/SudachiPy/pykakasi) available — kana "
            "targets cannot be produced. Run inside a container that ships one."
        )
        raise SystemExit(1)

    logger.info("=" * 70)
    logger.info("Whisper-encoder CTC fine-tuning (kana targets)")
    logger.info("=" * 70)
    logger.info(f"  Whisper     : {cfg['model']['name']}")
    logger.info(f"  Data dir    : {cfg['data']['data_dir']}")
    logger.info(f"  Output dir  : {out_dir}")

    # ---- datasets ----
    data_dir = cfg["data"]["data_dir"]
    train_ds = ConcatDataset(
        [
            CSJCorpusDataset(split=s, base_dir=data_dir)
            for s in cfg["data"]["train_splits"]
        ]
    )
    val_ds = ConcatDataset(
        [
            CSJCorpusDataset(split=s, base_dir=data_dir)
            for s in cfg["data"]["val_splits"]
        ]
    )
    logger.info(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # ---- vocab, feature extractor, tokenizer ----
    vocab_path = build_vocab(train_ds, cfg)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg["model"]["name"])

    train_ds = WhisperCTCDataset(train_ds, feature_extractor, tokenizer)
    val_ds = WhisperCTCDataset(val_ds, feature_extractor, tokenizer)

    # ---- model ----
    model = WhisperEncoderForCTC(
        cfg["model"]["name"],
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction=cfg["model"].get("ctc_loss_reduction", "mean"),
        ctc_zero_infinity=cfg["model"].get("ctc_zero_infinity", True),
        dropout=cfg["model"].get("dropout", 0.0),
    ).cuda()

    # Phase-1 freeze: only the CTC head trains until unfreeze_step.
    model.freeze_feature_encoder()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model loaded — {trainable:,} trainable / {total:,} total  "
        f"(CTC head only until step {cfg['freeze']['unfreeze_step']})"
    )

    # ---- training args ----
    ev, tr = cfg["eval"], cfg["training"]
    prec = cfg.get("precision", {})
    dl = cfg.get("dataloader", {})
    sched = cfg["schedule"]
    training_args = TrainingArguments(
        output_dir=out_dir,
        max_steps=sched["max_steps"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=cfg["optimizer"]["peak_lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        warmup_steps=sched["warmup_steps"],
        fp16=prec.get("fp16", False),
        bf16=prec.get("bf16", True),
        gradient_checkpointing=tr.get("gradient_checkpointing", False),
        eval_strategy="steps",
        eval_steps=ev["eval_steps"],
        logging_steps=ev["logging_steps"],
        save_steps=ev["save_steps"],
        save_strategy="steps",
        save_total_limit=ev.get("save_total_limit", 2),
        load_best_model_at_end=ev.get("load_best_model_at_end", True),
        metric_for_best_model=ev.get("metric_for_best_model", "cer"),
        greater_is_better=False,
        dataloader_num_workers=dl.get("num_workers", 4),
        dataloader_pin_memory=dl.get("pin_memory", True),
        report_to=cfg["output"].get("report_to", ["tensorboard"]),
        seed=cfg.get("seed", 42),
        remove_unused_columns=False,  # our forward takes input_features
    )

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["peak_lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=opt_cfg.get("eps", 1e-8),
    )
    scheduler = build_lr_scheduler(optimizer, cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=WhisperCTCCollator(tokenizer=tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=feature_extractor,
        compute_metrics=make_compute_metrics(tokenizer),
        optimizers=(optimizer, scheduler),
        callbacks=[UnfreezeCallback(cfg["freeze"]["unfreeze_step"])],
    )

    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=checkpoint)

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    feature_extractor.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper-encoder CTC fine-tuning (kana)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    return p.parse_args()


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> None:
    for item in overrides:
        key, _, val = item.partition("=")
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
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
    finally:
        logging.getLogger(__name__).info(f"Total time: {(time.time() - t0) / 3600:.2f}h")
