"""
Decode test sets (LibriSpeech or CSJ) with a CTC model: a finetuned WavLM-CTC,
any HF ``AutoModelForCTC`` checkpoint, or our Whisper-encoder kana CTC
(``src/finetune/whisper_ft_ctc.py``).

Produces a CTC pseudo-label file (TSV format):
    utt_id\tctc_decoded_text

This file is consumed by the inference script's "filter" mode to select
relevant hotwords from a bias list via n-gram indexing + Levenshtein scoring.

Usage:
    python scripts/decode_ctc.py \
        --wavlm-model exp/wavlm-ft-ctc-libri/final \
        --data-dir data/LibriSpeech \
        --output-dir exp/wavlm-ft-ctc-libri/ctc_decode \
        --splits test-clean test-other

    # Japanese (CSJ) with the Whisper kana-CTC model
    python pyscripts/decode_ctc.py \
        --wavlm-model exp/whisper-ft-ctc_ja/checkpoint-40000 \
        --base-model openai/whisper-medium \
        --data-dir data/csj --language ja \
        --output-dir exp/whisper-ft-ctc_ja/ctc_decode \
        --splits eval1_whisper_like_clean eval2_whisper_like_clean eval3_whisper_like_clean

"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    WavLMForCTC,
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.factory import DATASET_TYPES, build_dataset
from src.utils.compute_wer import compute_cer, score_transcripts
from src.utils.config_utils import setup_logging, log_separator
from src.utils.inference_utils import save_ctc_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load model from .pt checkpoint
# ---------------------------------------------------------------------------
def load_model_from_pt(
    pt_path: str,
    vocab_path: str,
    base_model: str = "microsoft/wavlm-large",
) -> Tuple[WavLMForCTC, Wav2Vec2Processor]:
    """
    Load a WavLM-CTC model from a .pt checkpoint file.

    Args:
        pt_path: Path to the .pt checkpoint (contains model_state_dict)
        vocab_path: Path to vocab.json for the CTC tokenizer
        base_model: HuggingFace model ID for the base architecture

    Returns:
        (model, processor)
    """
    # Build processor from vocab
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    vocab_size = len(tokenizer)
    logger.info(f"Vocab size from {vocab_path}: {vocab_size}")

    # Build model with matching vocab size
    model = WavLMForCTC.from_pretrained(
        base_model,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=vocab_size,
    )

    # Load checkpoint
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        info = {
            k: checkpoint[k]
            for k in ("epoch", "step", "wer", "loss")
            if k in checkpoint
        }
        logger.info(f"Checkpoint info: {info}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    logger.info(f"Loaded .pt checkpoint from {pt_path}")
    return model, processor


# ---------------------------------------------------------------------------
# Collate function for CTC decoding
# ---------------------------------------------------------------------------
def get_ctc_collate_fn(processor, device: str = "cuda", whisper: bool = False):
    """Create a collate function that featurises audio for the CTC model.

    ``processor`` is a Wav2Vec2-style processor (waveform ``input_values``) or,
    with ``whisper=True``, a ``WhisperFeatureExtractor`` (log-mel
    ``input_features`` padded to 30 s + a mel-frame attention mask).
    """

    def collate_fn(batch):
        audios = [sample["audio"] for sample in batch]
        inputs = processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            **({} if whisper else {"padding": True}),
        )
        key = "input_features" if whisper else "input_values"
        model_inputs = {key: inputs[key], "attention_mask": inputs["attention_mask"]}
        if device == "cuda":
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        return {
            "model_inputs": model_inputs,
            "utterance_id": [sample.get("utterance_id", "") for sample in batch],
            "transcription": [sample["transcription"] for sample in batch],
        }

    return collate_fn


# ---------------------------------------------------------------------------
# CTC greedy decode (character-level)
# ---------------------------------------------------------------------------
def ctc_greedy_decode(logits: torch.Tensor, processor) -> List[str]:
    """
    Greedy CTC decoding using the processor's vocabulary.

    Args:
        logits: (batch, seq_len, vocab_size) from WavLMForCTC
        processor: Wav2Vec2Processor or CTC tokenizer (anything with batch_decode)

    Returns:
        List of decoded strings
    """
    predicted_ids = torch.argmax(logits, dim=-1)
    decoded = processor.batch_decode(predicted_ids)
    return decoded


# ---------------------------------------------------------------------------
# Decode a full split
# ---------------------------------------------------------------------------
def kana_cer(hyps: List[str], refs: List[str]) -> float:
    """CER on katakana readings, so kana-only hypotheses (Whisper kana-CTC) are
    scored fairly against kanji+kana references. Needs a reading backend."""
    from src.finetune.whisper_ft_ctc import text_to_kana
    from src.utils.japanese import is_kana, normalize_reading

    def _hyp_kana(h: str) -> str:
        # Already kana: only normalise (re-analysing unsegmented kana can
        # change it); otherwise read kanji to kana like the reference.
        return normalize_reading(h) if is_kana(h) else text_to_kana(h)

    return compute_cer([_hyp_kana(h) for h in hyps], [text_to_kana(r) for r in refs])


@torch.no_grad()
def decode_split(
    model,
    processor,
    data_dir: str,
    split: str,
    batch_size: int = 8,
    num_workers: int = 4,
    data_device: str = "cuda",
    dataset_type: str = "auto",
    language: str = "en",
    feature_processor=None,
) -> Tuple[Dict[str, str], Dict[str, float], int]:
    """
    Decode all utterances in a split.

    ``processor`` decodes ids (``batch_decode``); ``feature_processor`` (if
    given) featurises audio — set for Whisper-CTC, where they differ.

    Returns:
        (results, metrics, num_samples) where results maps utt_id -> CTC text
        and metrics is ``{"wer": ...}`` (en) or
        ``{"cer_surface": ..., "cer_kana": ...}`` (ja)
    """
    dataset = build_dataset(data_dir, split, dataset_type, max_samples=0)
    logger.info(f"Loaded {len(dataset)} samples from {split}")

    collate_fn = get_ctc_collate_fn(
        feature_processor or processor,
        device=data_device,
        whisper=feature_processor is not None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )

    model.eval()
    results: Dict[str, str] = {}
    hyps = []
    refs = []

    for batch in tqdm(dataloader, desc=f"CTC decode [{split}]"):
        model_inputs = {k: v.to("cuda") for k, v in batch["model_inputs"].items()}
        logits = model(**model_inputs).logits

        decoded_texts = ctc_greedy_decode(logits, processor)

        for uid, text, ref in zip(
            batch["utterance_id"], decoded_texts, batch["transcription"]
        ):
            results[uid] = text.lower()
            hyps.append(text.lower())
            refs.append(ref.lower())
            logger.debug(f"  {uid}: CTC='{text}' | REF='{ref}'")

    metric, error_rate, _, _ = score_transcripts(hyps, refs, language=language)
    if language == "ja":
        # Surface CER is only meaningful for kanji+kana models; a kana-only CTC
        # model is judged by cer_kana (its surface CER is >100% by design).
        metrics = {"cer_surface": error_rate}
        from src.utils.japanese import has_reading_backend

        if has_reading_backend():
            metrics["cer_kana"] = kana_cer(hyps, refs)
        else:
            logger.warning("No reading backend: skipping kana CER")
    else:
        metrics = {metric: error_rate}
    logger.info(f"CTC decode metrics for {split}: {metrics}")
    return results, metrics, len(dataset)


def resolve_ctc_type(model_path: str, ctc_type: str = "auto") -> str:
    """``whisper`` for our Whisper-encoder CTC dirs (weights, no config.json)."""
    if ctc_type != "auto":
        return ctc_type
    p = Path(model_path)
    if p.is_dir() and not (p / "config.json").exists() and (
        (p / "whisper_ctc_config.json").exists() or (p / "model.safetensors").exists()
    ):
        return "whisper"
    return "hf"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode a dataset (LibriSpeech / CSJ) with a CTC model"
    )
    parser.add_argument(
        "--wavlm-model",
        type=str,
        default="exp/wavlm-ft-libri_2/final",
        help=(
            "CTC model: HF model dir/hub id, .pt checkpoint, or a Whisper-CTC "
            "final/checkpoint-* dir"
        ),
    )
    parser.add_argument(
        "--ctc-type",
        type=str,
        choices=["auto", "hf", "whisper"],
        default="auto",
        help=(
            "hf = AutoModelForCTC; whisper = Whisper-encoder kana CTC "
            "(auto: whisper if the dir has model weights but no config.json)"
        ),
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path to vocab.json (required when --wavlm-model is a .pt file)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/wavlm-large",
        help=(
            "Base HuggingFace model ID (used when --wavlm-model is a .pt file, or "
            "a Whisper-CTC checkpoint without whisper_ctc_config.json, e.g. "
            "openai/whisper-medium)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/LibriSpeech",
        help="Dataset root: LibriSpeech dir, or CSJ dump/raw dir",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_TYPES),
        default="auto",
        help="Dataset layout: librispeech, csj (Kaldi text/wav.scp), or auto-detect",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "ja"],
        default="en",
        help="Scoring: en = WER, ja = CER (+ kana-reading CER)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/wavlm-ft-libri_2/ctc_decode",
        help="Directory to save CTC decode files",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test-clean", "test-other"],
        help="Splits to decode",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(f"{args.output_dir}/decode.log")

    data_device = torch.device(args.data_device if torch.cuda.is_available() else "cpu")
    logger.info(f"Data loading device: {data_device}")

    # Load model
    logger.info(f"Loading CTC model from {args.wavlm_model}")
    feature_processor = None
    if resolve_ctc_type(args.wavlm_model, args.ctc_type) == "whisper":
        from src.finetune.whisper_ft_ctc import load_whisper_ctc

        # --base-model defaults to WavLM; only use it if it names a Whisper model.
        whisper_id = args.base_model if "whisper" in args.base_model else None
        model, processor, feature_processor = load_whisper_ctc(
            args.wavlm_model, vocab_path=args.vocab_path, whisper_id=whisper_id
        )
        model = model.to("cuda")
    elif args.wavlm_model.endswith(".pt"):
        if not args.vocab_path:
            raise ValueError("--vocab-path is required when using a .pt checkpoint")
        model, processor = load_model_from_pt(
            args.wavlm_model, args.vocab_path, args.base_model
        )
        model = model.to("cuda")
    else:
        # AutoModelForCTC / AutoProcessor so any CTC encoder works — WavLM,
        # HuBERT (e.g. Japanese kushinada-hubert), wav2vec2, …
        model = AutoModelForCTC.from_pretrained(args.wavlm_model).to("cuda")
        processor = AutoProcessor.from_pretrained(args.wavlm_model)
    model.eval()

    hidden = getattr(model.config, "hidden_size", getattr(model.config, "d_model", "?"))
    tokenizer = getattr(processor, "tokenizer", processor)
    logger.info(f"Model loaded. CTC vocab size: {len(tokenizer)}, Hidden size: {hidden}")

    # Decode each split
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {}

    for split in args.splits:
        log_separator()
        logger.info(f"Decoding split: {split}")

        results, metrics, num_samples = decode_split(
            model=model,
            processor=processor,
            data_dir=args.data_dir,
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_device=str(data_device),
            dataset_type=args.dataset,
            language=args.language,
            feature_processor=feature_processor,
        )

        out_path = os.path.join(args.output_dir, f"{split}_ctc.tsv")
        save_ctc_file(results, out_path)
        summary[split] = {**metrics, "num_samples": num_samples}

        logger.info(f"  Decoded {len(results)} utterances -> {out_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
