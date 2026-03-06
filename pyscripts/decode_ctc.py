"""
Decode LibriSpeech test sets using the finetuned WavLM-CTC model.

Produces a CTC pseudo-label file (TSV format):
    utt_id\tctc_decoded_text

This file is consumed by the inference script's "filter" mode to select
relevant hotwords from a bias list via n-gram indexing + Levenshtein scoring.

Usage:
    python scripts/decode_ctc.py \
        --wavlm-model exp/wavlm-ft-libri/final \
        --data-dir data/LibriSpeech \
        --output-dir exp/wavlm-ft-libri/ctc_decode \
        --splits test-clean test-other

"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WavLMForCTC, Wav2Vec2Processor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.librispeech_dataset import LibriSpeechDataset
from src.utils.compute_wer import compute_wer
from src.utils.config_utils import setup_logging, log_separator

logger = logging.getLogger(__name__)


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
    device: str = "cuda",
) -> Dict[str, str]:
    """
    Decode all utterances in a LibriSpeech split.

    Returns:
        dict mapping utt_id -> CTC decoded text
    """
    dataset = LibriSpeechDataset(data_path=data_dir, split=split)
    logger.info(f"Loaded {len(dataset)} samples from {split}")

    collate_fn = get_ctc_collate_fn(processor, device=device)
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
    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_ctc_file(results: Dict[str, str], output_path: str) -> None:
    """Save CTC decode results as TSV: utt_id\\tctc_text"""
    with open(output_path, "w", encoding="utf-8") as f:
        for uid, text in sorted(results.items()):
            f.write(f"{uid}\t{text}\n")
    logger.info(f"Saved {len(results)} CTC decodes to {output_path}")


def load_ctc_file(ctc_path: str) -> Dict[str, str]:
    """Load CTC decode results from TSV file."""
    results: Dict[str, str] = {}
    with open(ctc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                results[parts[0]] = parts[1]
    return results


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
        help="Path to finetuned WavLM-CTC model directory",
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

    device = torch.device(args.data_device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading WavLM-CTC model from {args.wavlm_model}")
    model = WavLMForCTC.from_pretrained(args.wavlm_model).to(device)
    processor = Wav2Vec2Processor.from_pretrained(args.wavlm_model)
    model.eval()

    logger.info(
        f"Model loaded. Vocab size: {model.config.vocab_size}, "
        f"Hidden size: {model.config.hidden_size}"
    )

    # Decode each split
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        log_separator()
        logger.info(f"Decoding split: {split}")

        results = decode_split(
            model=model,
            processor=processor,
            data_dir=args.data_dir,
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=str(device),
        )

        out_path = os.path.join(args.output_dir, f"{split}_ctc.tsv")
        save_ctc_file(results, out_path)

        logger.info(f"  Decoded {len(results)} utterances -> {out_path}")


if __name__ == "__main__":
    main()
