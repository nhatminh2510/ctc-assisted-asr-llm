"""
Decode LibriSpeech test sets using the finetuned WavLM-CTC model.

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
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.librispeech_dataset import LibriSpeechDataset
from src.utils.dataset_utils import to_cuda
from src.utils.compute_wer import compute_wer
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
def get_ctc_collate_fn(processor: Wav2Vec2Processor, device: str = "cuda"):
    """Create a collate function that processes audio for WavLM-CTC."""

    def collate_fn(batch):
        audios = [sample["audio"] for sample in batch]
        inputs = processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        if device == "cuda":
            inputs = inputs.to(device)

        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "utterance_id": [sample.get("utterance_id", "") for sample in batch],
            "transcription": [sample["transcription"] for sample in batch],
        }

    return collate_fn


# ---------------------------------------------------------------------------
# CTC greedy decode (character-level)
# ---------------------------------------------------------------------------
def ctc_greedy_decode(logits: torch.Tensor, processor: Wav2Vec2Processor) -> List[str]:
    """
    Greedy CTC decoding using the processor's vocabulary.

    Args:
        logits: (batch, seq_len, vocab_size) from WavLMForCTC
        processor: Wav2Vec2Processor with tokenizer

    Returns:
        List of decoded strings
    """
    predicted_ids = torch.argmax(logits, dim=-1)
    decoded = processor.batch_decode(predicted_ids)
    return decoded


# ---------------------------------------------------------------------------
# Decode a full split
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_split(
    model: WavLMForCTC,
    processor: Wav2Vec2Processor,
    data_dir: str,
    split: str,
    batch_size: int = 8,
    num_workers: int = 4,
    data_device: str = "cuda",
) -> Tuple[Dict[str, str], float, int]:
    """
    Decode all utterances in a LibriSpeech split.

    Returns:
        (results, wer, num_samples) where results maps utt_id -> CTC decoded text
    """
    dataset = LibriSpeechDataset(data_path=data_dir, split=split)
    logger.info(f"Loaded {len(dataset)} samples from {split}")

    collate_fn = get_ctc_collate_fn(processor, device=data_device)
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
        if data_device != "cuda":
            batch = to_cuda(batch)
        logits = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
        ).logits

        decoded_texts = ctc_greedy_decode(logits, processor)

        for uid, text, ref in zip(
            batch["utterance_id"], decoded_texts, batch["transcription"]
        ):
            results[uid] = text.lower()
            hyps.append(text.lower())
            refs.append(ref.lower())
            logger.debug(f"  {uid}: CTC='{text}' | REF='{ref}'")
    wer = compute_wer(hyps, refs)
    logger.info(f"CTC decode WER for {split}: {wer}")
    return results, wer, len(dataset)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode LibriSpeech using finetuned WavLM-CTC"
    )
    parser.add_argument(
        "--wavlm-model",
        type=str,
        default="exp/wavlm-ft-libri_2/final",
        help="Path to finetuned WavLM-CTC model directory or .pt checkpoint",
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
        help="Base HuggingFace model ID (used when --wavlm-model is a .pt file)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/LibriSpeech",
        help="Root directory of LibriSpeech data",
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
        help="LibriSpeech splits to decode",
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
    logger.info(f"Loading WavLM-CTC model from {args.wavlm_model}")
    if args.wavlm_model.endswith(".pt"):
        if not args.vocab_path:
            raise ValueError("--vocab-path is required when using a .pt checkpoint")
        model, processor = load_model_from_pt(
            args.wavlm_model, args.vocab_path, args.base_model
        )
        model = model.to("cuda")
    else:
        model = WavLMForCTC.from_pretrained(args.wavlm_model).to("cuda")
        processor = Wav2Vec2Processor.from_pretrained(args.wavlm_model)
    model.eval()

    logger.info(
        f"Model loaded. Vocab size: {model.config.vocab_size}, "
        f"Hidden size: {model.config.hidden_size}"
    )

    # Decode each split
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {}

    for split in args.splits:
        log_separator()
        logger.info(f"Decoding split: {split}")

        results, wer, num_samples = decode_split(
            model=model,
            processor=processor,
            data_dir=args.data_dir,
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_device=str(data_device),
        )

        out_path = os.path.join(args.output_dir, f"{split}_ctc.tsv")
        save_ctc_file(results, out_path)
        summary[split] = {"wer": wer, "num_samples": num_samples}

        logger.info(f"  Decoded {len(results)} utterances -> {out_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
