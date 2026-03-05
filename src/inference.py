"""
Inference script for CTC-Assisted LLM-Based Contextual ASR.

Three inference modes (matching the original paper):
  - nobias:  No hotwords. Prompt = "Transcribe speech to text."
  - gt:      Ground-truth hotwords from bias list (oracle / upper bound).
  - filter:  CTC-based filtering — use CTC pseudo-labels + bias list +
             HotwordFilter to select relevant hotwords.

Also supports single-file transcription.

Based on: https://arxiv.org/pdf/2411.06437
Reference: https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/contextual_asr
"""

import os
import sys
import json
import time
import torch
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import CTCAssistedLLMASR
from src.dataset import LibriSpeechDataset
from src.hotword_filter import HotwordFilter, create_prompt_with_hotwords
from src.utils import (
    load_audio,
    load_config,
    load_checkpoint,
    compute_wer,
    setup_logging,
    set_seed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bias list / CTC file loaders
# ---------------------------------------------------------------------------
def load_bias_list(bias_list_json: str) -> Dict[str, Dict]:
    """
    Load per-utterance bias list JSON.

    Returns a dict: utt_id -> {
        "bias_list": [...],
        "true_hotwords": [...],
        "transcription": "...",
    }
    """
    with open(bias_list_json, "r") as f:
        data = json.load(f)
    result = {}
    for entry in data:
        uid = entry["utt_id"]
        result[uid] = {
            "bias_list": entry.get("bias_list", []),
            "true_hotwords": entry.get("true_hotwords", []),
            "transcription": entry.get("transcription", ""),
        }
    logger.info(f"Loaded bias list for {len(result)} utterances from {bias_list_json}")
    return result


def load_bias_lists(paths: List[str]) -> Dict[str, Dict]:
    """Load and merge multiple bias list JSONs (one per split)."""
    merged: Dict[str, Dict] = {}
    for p in paths:
        merged.update(load_bias_list(p))
    logger.info(
        f"Merged bias lists: {len(merged)} utterances from {len(paths)} file(s)"
    )
    return merged


def load_ctc_file(ctc_path: str) -> Dict[str, str]:
    """
    Load CTC pseudo-label file (TSV: utt_id\\tctc_text).

    Returns dict: utt_id -> ctc_decoded_text
    """
    results: Dict[str, str] = {}
    with open(ctc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                results[parts[0]] = parts[1]
    logger.info(f"Loaded {len(results)} CTC decodes from {ctc_path}")
    return results


def load_ctc_files(paths: List[str]) -> Dict[str, str]:
    """Load and merge multiple CTC decode files (one per split)."""
    merged: Dict[str, str] = {}
    for p in paths:
        merged.update(load_ctc_file(p))
    logger.info(f"Merged CTC files: {len(merged)} utterances from {len(paths)} file(s)")
    return merged


# ---------------------------------------------------------------------------
# Collate functions for each inference mode
# ---------------------------------------------------------------------------
def get_nobias_collate_fn(encoder_processor, decoder_processor, device: str = "cuda"):
    """
    Collate for nobias mode: no hotwords in prompt.
    Prompt: "USER: Transcribe speech to text.\\n ASSISTANT:"
    """
    prompt_text = create_prompt_with_hotwords([])

    def collate_fn(batch):
        encoder_inputs = encoder_processor(
            [sample["audio"] for sample in batch],
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        if device == "cuda":
            encoder_inputs = encoder_inputs.to(device)

        prompt_inputs = decoder_processor(
            [prompt_text] * len(batch),
            return_tensors="pt",
            padding=True,
        )
        if device == "cuda":
            prompt_inputs = prompt_inputs.to(device)

        return {
            "input_values": encoder_inputs.input_values,
            "encoder_attention_mask": encoder_inputs.attention_mask,
            "generate_input_ids": prompt_inputs.input_ids,
            "generate_attention_mask": prompt_inputs.attention_mask,
            "transcription": [s["transcription"] for s in batch],
            "utterance_id": [s.get("utterance_id", "") for s in batch],
        }

    return collate_fn


def get_gt_collate_fn(
    encoder_processor,
    decoder_processor,
    bias_data: Dict[str, Dict],
    device: str = "cuda",
):
    """
    Collate for gt (ground-truth) mode: use true_hotwords from bias list.
    Each utterance gets its own per-utterance prompt (no batch sharing).
    """

    def collate_fn(batch):
        encoder_inputs = encoder_processor(
            [sample["audio"] for sample in batch],
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        if device == "cuda":
            encoder_inputs = encoder_inputs.to(device)

        # Per-utterance GT hotwords
        prompts = []
        for sample in batch:
            uid = sample.get("utterance_id", "")
            if uid in bias_data:
                hw = bias_data[uid]["true_hotwords"]
            else:
                hw = []
            prompts.append(create_prompt_with_hotwords(hw))

        prompt_inputs = decoder_processor(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        if device == "cuda":
            prompt_inputs = prompt_inputs.to(device)

        return {
            "input_values": encoder_inputs.input_values,
            "encoder_attention_mask": encoder_inputs.attention_mask,
            "generate_input_ids": prompt_inputs.input_ids,
            "generate_attention_mask": prompt_inputs.attention_mask,
            "transcription": [s["transcription"] for s in batch],
            "utterance_id": [s.get("utterance_id", "") for s in batch],
        }

    return collate_fn


def get_filter_collate_fn(
    encoder_processor,
    decoder_processor,
    bias_data: Dict[str, Dict],
    ctc_data: Dict[str, str],
    hotword_filter: HotwordFilter,
    device: str = "cuda",
):
    """
    Collate for filter mode: use CTC pseudo-labels + HotwordFilter to select
    relevant hotwords from the bias list for each utterance.
    """

    def collate_fn(batch):
        encoder_inputs = encoder_processor(
            [sample["audio"] for sample in batch],
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        if device == "cuda":
            encoder_inputs = encoder_inputs.to(device)

        # Per-utterance filtered hotwords
        prompts = []
        for sample in batch:
            uid = sample.get("utterance_id", "")
            ctc_text = ctc_data.get(uid, "")
            bias_list = bias_data.get(uid, {}).get("bias_list", [])

            if ctc_text and bias_list:
                filtered_hw = hotword_filter.filter_hotwords(ctc_text, bias_list)
            else:
                filtered_hw = []

            prompts.append(create_prompt_with_hotwords(filtered_hw))

        prompt_inputs = decoder_processor(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        if device == "cuda":
            prompt_inputs = prompt_inputs.to(device)

        return {
            "input_values": encoder_inputs.input_values,
            "encoder_attention_mask": encoder_inputs.attention_mask,
            "generate_input_ids": prompt_inputs.input_ids,
            "generate_attention_mask": prompt_inputs.attention_mask,
            "transcription": [s["transcription"] for s in batch],
            "utterance_id": [s.get("utterance_id", "") for s in batch],
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_dataset(
    model: CTCAssistedLLMASR,
    dataloader: DataLoader,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    do_sample: bool = False,
    data_device: str = "cuda",
) -> Dict[str, Any]:
    """Run inference over a DataLoader and compute WER.

    Returns a dict with:
        wer       – float, overall WER
        hyps      – list[str], normalised hypotheses
        refs      – list[str], normalised references
        raw_hyps  – list[str], raw hypotheses (before normalisation)
        raw_refs  – list[str], raw references
        utt_ids   – list[str], utterance ids
    """

    def to_cuda(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Đẩy tất cả Tensor trong batch lên device, giữ nguyên các kiểu dữ liệu khác."""
        return {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    normalizer = BasicTextNormalizer()
    model.eval()

    raw_hyps: List[str] = []
    raw_refs: List[str] = []
    utt_ids: List[str] = []

    for batch in tqdm(dataloader, desc="Inference"):
        if data_device != "cuda":
            batch = to_cuda(batch)

        generated_ids = model.generate(
            input_values=batch["input_values"],
            input_ids=batch["generate_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_attention_mask=batch["generate_attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )

        predicted_texts = model.decoder_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        raw_hyps.extend(predicted_texts)
        raw_refs.extend(batch["transcription"])
        utt_ids.extend(batch["utterance_id"])

        # Log a few examples per batch
        for hyp, ref in zip(predicted_texts, batch["transcription"]):
            logger.debug(f"HYP: {hyp}")
            logger.debug(f"REF: {ref}")

    # Normalise
    hyps = [normalizer(h) for h in raw_hyps]
    refs = [normalizer(r) for r in raw_refs]

    wer_score = compute_wer(hyps, refs) if hyps and refs else 0.0

    return {
        "wer": wer_score,
        "hyps": hyps,
        "refs": refs,
        "raw_hyps": raw_hyps,
        "raw_refs": raw_refs,
        "utt_ids": utt_ids,
    }


# ---------------------------------------------------------------------------
# Single-file transcription helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def transcribe_single(
    model: CTCAssistedLLMASR,
    audio_path: str,
    device: torch.device,
    hotwords: Optional[List[str]] = None,
    max_new_tokens: int = 256,
    num_beams: int = 4,
) -> str:
    """Transcribe a single audio file and return the text."""
    model.eval()

    audio = load_audio(audio_path).to(device)
    encoder_inputs = model.encoder_processor(
        audio.squeeze(0).cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    prompt = create_prompt_with_hotwords(hotwords or [])
    prompt_inputs = model.decoder_processor(
        prompt, return_tensors="pt", padding=True
    ).to(device)

    generated_ids = model.generate(
        input_values=encoder_inputs.input_values,
        input_ids=prompt_inputs.input_ids,
        encoder_attention_mask=encoder_inputs.attention_mask,
        decoder_attention_mask=prompt_inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    transcription = model.decoder_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return transcription


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------
def save_results(results: Dict[str, Any], output_dir: str, split: str) -> None:
    """Persist evaluation results to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Summary JSON
    summary = {
        "split": split,
        "wer": results["wer"],
        "num_samples": len(results["hyps"]),
    }
    summary_path = out / f"{split}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # 2. Per-utterance details
    details = []
    for uid, hyp, ref, raw_hyp, raw_ref in zip(
        results["utt_ids"],
        results["hyps"],
        results["refs"],
        results["raw_hyps"],
        results["raw_refs"],
    ):
        details.append(
            {
                "utt_id": uid,
                "hyp": hyp,
                "ref": ref,
                "raw_hyp": raw_hyp,
                "raw_ref": raw_ref,
            }
        )
    details_path = out / f"{split}_details.json"
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    logger.info(f"Details saved to {details_path}")

    # 3. Plain text hyp/ref files (for external scoring tools)
    with open(out / f"{split}_hyp.txt", "w") as fh, open(
        out / f"{split}_ref.txt", "w"
    ) as fr:
        for uid, hyp, ref in zip(results["utt_ids"], results["hyps"], results["refs"]):
            fh.write(f"{uid} {hyp}\n")
            fr.write(f"{uid} {ref}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="CTC-Assisted LLM-Based Contextual ASR – Inference"
    )

    # --- mode selection ---
    parser.add_argument(
        "--mode",
        choices=["eval", "transcribe"],
        default="eval",
        help="'eval' – evaluate on LibriSpeech test splits. 'transcribe' – single file.",
    )
    parser.add_argument(
        "--infer-type",
        choices=["nobias", "gt", "filter"],
        default="nobias",
        help=(
            "Inference type (for eval mode):\n"
            "  nobias – no hotwords in prompt\n"
            "  gt     – ground-truth hotwords from bias list\n"
            "  filter – CTC-filtered hotwords from bias list"
        ),
    )

    # --- paths ---
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: <exp-dir>/training_config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/LibriSpeech",
        help="Root directory of LibriSpeech data (for eval mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: <exp-dir>/inference/<infer_type>/)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (for transcribe mode)",
    )

    # --- dataset splits ---
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test-clean", "test-other"],
        help="LibriSpeech splits to evaluate",
    )

    # --- model ---
    parser.add_argument(
        "--wavlm-model",
        type=str,
        default=None,
        help="WavLM model id or path (overrides config)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model id or path (overrides config)",
    )

    # --- generation ---
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--do-sample", action="store_true")

    # --- data ---
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    # --- bias list / CTC file (for gt and filter modes) ---
    parser.add_argument(
        "--bias-list-json",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Path(s) to per-utterance bias list JSON(s). "
            "Pass one per split, e.g.: "
            "--bias-list-json data/hotwords/test-clean_bias_list.json "
            "data/hotwords/test-other_bias_list.json"
        ),
    )
    parser.add_argument(
        "--ctc-file",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Path(s) to CTC pseudo-label file(s). "
            "Pass one per split, e.g.: "
            "--ctc-file data/ctc_decode/test-clean_ctc.tsv "
            "data/ctc_decode/test-other_ctc.tsv"
        ),
    )

    # --- hotword filter params (for filter mode) ---
    parser.add_argument(
        "--common-words-path",
        type=str,
        default=None,
        help="Path to common_words_5k.txt (overrides config)",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=None,
        help="Levenshtein.ratio threshold for filter mode (default: 0.9)",
    )
    parser.add_argument(
        "--word-num",
        type=int,
        default=None,
        help="Max hotwords to keep after filtering (default: 15)",
    )

    # --- hotwords for transcribe mode ---
    parser.add_argument(
        "--hotwords",
        type=str,
        nargs="+",
        default=None,
        help="Explicit hotword list for transcribe mode",
    )

    # --- misc ---
    parser.add_argument("--data-device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    exp_dir = ckpt_path.parent.parent  # <exp>/checkpoints/<ckpt>.pt -> <exp>

    if args.output_dir is None:
        args.output_dir = str(exp_dir / "inference" / args.infer_type)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(
        log_file=str(Path(args.output_dir) / "inference.log"), level=logging.INFO
    )

    logger.info(f"Inference mode: {args.mode}, infer_type: {args.infer_type}")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config_path = args.config or str(exp_dir / "training_config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path}, using defaults.")
        config = {}

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    infer_cfg = config.get("inference", {})
    filter_cfg = config.get("hotword_filter", {})

    # CLI overrides
    wavlm_id = args.wavlm_model or model_cfg.get(
        "wavlm_model_path", "facebook/wavlm-large"
    )
    llm_id = args.llm_model or model_cfg.get("llm_model_path", "lmsys/vicuna-7b-v1.5")
    max_new_tokens = args.max_new_tokens or infer_cfg.get("max_new_tokens", 256)
    num_beams = args.num_beams or infer_cfg.get("num_beams", 4)

    # ------------------------------------------------------------------
    # Validate mode-specific requirements
    # ------------------------------------------------------------------
    if args.mode == "eval" and args.infer_type in ("gt", "filter"):
        if not args.bias_list_json:
            logger.error(
                f"--bias-list-json is required for infer_type='{args.infer_type}'"
            )
            sys.exit(1)

    if args.mode == "eval" and args.infer_type == "filter":
        if not args.ctc_file:
            logger.error("--ctc-file is required for infer_type='filter'")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Load bias list / CTC file
    # ------------------------------------------------------------------
    bias_data: Optional[Dict[str, Dict]] = None
    ctc_data: Optional[Dict[str, str]] = None

    if args.bias_list_json:
        bias_data = load_bias_lists(args.bias_list_json)

    if args.ctc_file:
        ctc_data = load_ctc_files(args.ctc_file)

    # ------------------------------------------------------------------
    # Build HotwordFilter (for filter mode)
    # ------------------------------------------------------------------
    hotword_filter: Optional[HotwordFilter] = None
    if args.infer_type == "filter":
        common_words_path = args.common_words_path or filter_cfg.get(
            "common_words_path", "data/common_words_5k.txt"
        )
        prob_threshold = (
            args.probability_threshold
            if args.probability_threshold is not None
            else filter_cfg.get("probability_threshold", 0.9)
        )
        word_num = (
            args.word_num
            if args.word_num is not None
            else filter_cfg.get("word_num", 15)
        )
        n_gram = filter_cfg.get("n_gram", 2)

        hotword_filter = HotwordFilter(
            common_words_path=common_words_path,
            n_gram=n_gram,
            word_num=word_num,
            probability_threshold=prob_threshold,
            per_word_scoring=True,
        )
        logger.info(
            f"HotwordFilter: n_gram={n_gram}, word_num={word_num}, "
            f"threshold={prob_threshold}, common_words={common_words_path}"
        )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    device = torch.device(args.data_device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading model – encoder: {wavlm_id}, LLM: {llm_id}")
    model = CTCAssistedLLMASR(
        encoder_id=wavlm_id,
        decoder_id=llm_id,
        freeze_encoder=True,
        freeze_llm=True,
    )

    logger.info(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(model, str(ckpt_path), device=str(device))
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Single-file transcribe mode
    # ------------------------------------------------------------------
    if args.mode == "transcribe":
        if not args.audio:
            logger.error("--audio is required in transcribe mode")
            sys.exit(1)
        text = transcribe_single(
            model,
            args.audio,
            device,
            hotwords=args.hotwords,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        logger.info(f"\nTranscription: {text}\n")
        out = {"audio": args.audio, "transcription": text}
        if args.hotwords:
            out["hotwords"] = args.hotwords
        out_path = Path(args.output_dir) / "transcription.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to {out_path}")
        return

    # ------------------------------------------------------------------
    # Eval mode – evaluate on LibriSpeech test splits
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    overall_hyps: List[str] = []
    overall_refs: List[str] = []
    split_wers: Dict[str, float] = {}

    t0 = time.time()

    for split in args.splits:
        logger.info("\n")
        logger.info(f"Evaluating: split={split}, infer_type={args.infer_type}")

        dataset = LibriSpeechDataset(
            data_path=str(data_dir), split=split, config=data_cfg
        )
        logger.info(f"Loaded {len(dataset)} samples from {split}")

        # Select collate function based on infer_type
        if args.infer_type == "nobias":
            collate_fn = get_nobias_collate_fn(
                model.encoder_processor,
                model.decoder_processor,
                device=str(device),
            )
        elif args.infer_type == "gt":
            collate_fn = get_gt_collate_fn(
                model.encoder_processor,
                model.decoder_processor,
                bias_data=bias_data,
                device=str(device),
            )
        elif args.infer_type == "filter":
            collate_fn = get_filter_collate_fn(
                model.encoder_processor,
                model.decoder_processor,
                bias_data=bias_data,
                ctc_data=ctc_data,
                hotword_filter=hotword_filter,
                device=str(device),
            )
        else:
            raise ValueError(f"Unknown infer_type: {args.infer_type}")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=(args.num_workers > 0),
        )

        results = evaluate_dataset(
            model,
            dataloader,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=args.do_sample,
            data_device=args.data_device,
        )

        split_wers[split] = results["wer"]
        overall_hyps.extend(results["hyps"])
        overall_refs.extend(results["refs"])

        logger.info(
            f"[{split}] WER = {results['wer']*100:.2f}%  "
            f"({len(results['hyps'])} utterances)"
        )

        # Save per-split results
        save_results(results, args.output_dir, split)

    elapsed = time.time() - t0

    # Overall WER (if multiple splits)
    if len(args.splits) > 1:
        overall_wer = compute_wer(overall_hyps, overall_refs)
        split_wers["overall"] = overall_wer
        logger.info(f"\nOverall WER = {overall_wer*100:.2f}%")

    # Print summary
    logger.info("\n")
    logger.info(f"INFERENCE RESULTS  (infer_type={args.infer_type})")
    for split, w in split_wers.items():
        logger.info(f"  {split:20s}  WER = {w*100:.2f}%")
    logger.info(f"  {'time':20s}  {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save global summary
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "infer_type": args.infer_type,
                "splits": args.splits,
                "wer": split_wers,
                "num_samples": len(overall_hyps),
                "elapsed_seconds": elapsed,
                "config": {
                    "max_new_tokens": max_new_tokens,
                    "num_beams": num_beams,
                    "batch_size": args.batch_size,
                },
            },
            f,
            indent=2,
        )
    logger.info(f"Global summary saved to {summary_path}")


if __name__ == "__main__":
    main()
