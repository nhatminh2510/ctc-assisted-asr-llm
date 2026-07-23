"""
Inference script for CTC-Assisted LLM-Based Contextual ASR.

Three inference modes (matching the original paper):
  - nobias:  No hotwords. Prompt = "Transcribe speech to text."
  - gt:      Ground-truth hotwords from bias list (oracle / upper bound).
  - filter:  CTC-based filtering — use CTC pseudo-labels + bias list +
             HotwordFilter to select relevant hotwords.

Also supports single-file transcription.

"""

import os
import sys
import json
import time
import torch
import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.model import CTCAssistedLLMASR
from src.dataset.librispeech_dataset import LibriSpeechDataset
from src.utils.create_prompt import create_prompt_with_hotwords
from src.utils.dataset_utils import load_audio
from src.utils.config_utils import load_config, setup_logging, set_seed, log_separator
from src.utils.train_utils import load_checkpoint
from src.utils.compute_wer import compute_wer
from src.utils.inference_utils import load_bias_lists, load_ctc_files, save_results

if TYPE_CHECKING:
    from src.inference.hotword_filter_F1 import HotwordFilter

logger = logging.getLogger(__name__)

# Filter method defaults (from paper Table 4; "ja" is the Japanese filter)
FILTER_DEFAULTS = {
    "F1": {"probability_threshold": 0.95, "word_num": 5},
    "F2": {"probability_threshold": 0.9, "word_num": 15},
    "F3": {"probability_threshold": 0.9, "word_num": 15},
    "ja": {"probability_threshold": 0.5, "word_num": 15},
}


def load_hotword_filter(method: str):
    """Dynamically import the HotwordFilter class for the given method."""
    if method == "F1":
        from src.inference.hotword_filter_F1 import HotwordFilter
    elif method == "F2":
        from src.inference.hotword_filter_F2 import HotwordFilter
    elif method == "F3":
        from src.inference.hotword_filter_F3 import HotwordFilter
    elif method == "ja":
        from src.inference.hotword_filter_ja import HotwordFilter
    else:
        raise ValueError(
            f"Unknown filter method: {method}. Choose from F1, F2, F3, ja."
        )
    return HotwordFilter


# ---------------------------------------------------------------------------
# Collate functions for each inference mode
# ---------------------------------------------------------------------------
def get_nobias_collate_fn(
    encoder, decoder_processor, device: str = "cuda", language: str = "en"
):
    """
    Collate for nobias mode: no hotwords in prompt.
    Prompt: "USER: Transcribe speech to text.\\n ASSISTANT:"
    """
    prompt_text = create_prompt_with_hotwords([], language=language)
    move_to = device if device == "cuda" else None

    def collate_fn(batch):
        encoder_inputs = encoder.process_audio(
            [sample["audio"] for sample in batch], device=move_to
        )

        prompt_inputs = decoder_processor(
            [prompt_text] * len(batch),
            return_tensors="pt",
            padding=True,
        )
        if move_to:
            prompt_inputs = prompt_inputs.to(device)

        return {
            "input_values": encoder_inputs["input_values"],
            "encoder_attention_mask": encoder_inputs["encoder_attention_mask"],
            "generate_input_ids": prompt_inputs.input_ids,
            "generate_attention_mask": prompt_inputs.attention_mask,
            "transcription": [s["transcription"] for s in batch],
            "utterance_id": [s.get("utterance_id", "") for s in batch],
        }

    return collate_fn


def get_gt_collate_fn(
    encoder,
    decoder_processor,
    bias_data: Dict[str, Dict],
    device: str = "cuda",
    language: str = "en",
    reading_fn=None,
):
    """
    Collate for gt (ground-truth) mode: use true_hotwords from bias list.
    Each utterance gets its own per-utterance prompt (no batch sharing).

    ``reading_fn`` (optional) maps a surface hotword to its reading; when given
    (Japanese) the prompt annotates each hotword as ``橋（ハシ）``.
    """
    move_to = device if device == "cuda" else None

    def collate_fn(batch):
        encoder_inputs = encoder.process_audio(
            [sample["audio"] for sample in batch], device=move_to
        )

        # Per-utterance GT hotwords
        prompts = []
        for sample in batch:
            uid = sample.get("utterance_id", "")
            hw = bias_data[uid]["true_hotwords"] if uid in bias_data else []
            readings = [reading_fn(w) for w in hw] if reading_fn else None
            prompts.append(
                create_prompt_with_hotwords(hw, language=language, readings=readings)
            )

        prompt_inputs = decoder_processor(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        if move_to:
            prompt_inputs = prompt_inputs.to(device)

        return {
            "input_values": encoder_inputs["input_values"],
            "encoder_attention_mask": encoder_inputs["encoder_attention_mask"],
            "generate_input_ids": prompt_inputs.input_ids,
            "generate_attention_mask": prompt_inputs.attention_mask,
            "transcription": [s["transcription"] for s in batch],
            "utterance_id": [s.get("utterance_id", "") for s in batch],
        }

    return collate_fn


def get_filter_collate_fn(
    encoder,
    decoder_processor,
    bias_data: Dict[str, Dict],
    ctc_data: Dict[str, str],
    hotword_filter: "HotwordFilter",
    device: str = "cuda",
    language: str = "en",
):
    """
    Collate for filter mode: use CTC pseudo-labels + HotwordFilter to select
    relevant hotwords from the bias list for each utterance.

    For Japanese (``language == "ja"``) the filter also returns each hotword's
    reading, which is injected into the prompt as ``橋（ハシ）`` so the LLM can
    disambiguate homophones the (kana) CTC hypothesis cannot.
    """
    move_to = device if device == "cuda" else None
    use_readings = language == "ja" and hasattr(
        hotword_filter, "filter_hotwords_with_readings"
    )

    def collate_fn(batch):
        encoder_inputs = encoder.process_audio(
            [sample["audio"] for sample in batch], device=move_to
        )

        # Per-utterance filtered hotwords
        prompts = []
        for sample in batch:
            uid = sample.get("utterance_id", "")
            ctc_text = ctc_data.get(uid, "")
            bias_list = bias_data.get(uid, {}).get("bias_list", [])

            readings = None
            if ctc_text and bias_list:
                if use_readings:
                    pairs = hotword_filter.filter_hotwords_with_readings(
                        ctc_text, bias_list
                    )
                    filtered_hw = [s for s, _ in pairs]
                    readings = [r for _, r in pairs]
                else:
                    filtered_hw = hotword_filter.filter_hotwords(ctc_text, bias_list)
            else:
                filtered_hw = []

            prompts.append(
                create_prompt_with_hotwords(
                    filtered_hw, language=language, readings=readings
                )
            )

        prompt_inputs = decoder_processor(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        if move_to:
            prompt_inputs = prompt_inputs.to(device)

        return {
            "input_values": encoder_inputs["input_values"],
            "encoder_attention_mask": encoder_inputs["encoder_attention_mask"],
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
    hyps = [normalizer(h.strip()) for h in raw_hyps]
    refs = [normalizer(r.strip()) for r in raw_refs]

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
    language: str = "en",
) -> str:
    """Transcribe a single audio file and return the text."""
    model.eval()

    audio = load_audio(audio_path)
    encoder_inputs = model.encoder.process_audio(
        [audio.squeeze(0).cpu().numpy()], device=str(device)
    )

    prompt = create_prompt_with_hotwords(hotwords or [], language=language)
    prompt_inputs = model.decoder_processor(
        prompt, return_tensors="pt", padding=True
    ).to(device)

    generated_ids = model.generate(
        input_values=encoder_inputs["input_values"],
        input_ids=prompt_inputs.input_ids,
        encoder_attention_mask=encoder_inputs["encoder_attention_mask"],
        decoder_attention_mask=prompt_inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    transcription = model.decoder_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return transcription


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
        "--infer-config",
        type=str,
        default=None,
        help="Path to inference config YAML (default: config/inference.yaml)",
    )
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
        help="Encoder model id or path (overrides config)",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default=None,
        choices=["auto", "wavlm", "hubert", "wav2vec2", "whisper"],
        help="Speech encoder family (default: auto-detect from the checkpoint)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model id or path (overrides config)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["en", "ja"],
        help="Prompt/filter language (default: from config or 'en')",
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
        "--filter-method",
        type=str,
        choices=["F1", "F2", "F3", "ja"],
        default="F3",
        help=(
            "Hotword filtering method (for filter mode):\n"
            "  F1 – no common word removal, threshold=0.95, max=5\n"
            "  F2 – common word removal, whole-sentence scoring, threshold=0.9\n"
            "  F3 – common word removal, per-word best-match, no threshold (default)\n"
            "  ja – Japanese reading-based filter (segmentation + homophone-aware)"
        ),
    )
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
        help="Levenshtein.ratio threshold for filter mode (default depends on method)",
    )
    parser.add_argument(
        "--word-num",
        type=int,
        default=None,
        help="Max hotwords to keep after filtering (default depends on method)",
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
    training_config_path = args.config or str(exp_dir / "training_config.yaml")
    if os.path.exists(training_config_path):
        config = load_config(training_config_path)
        logger.info(f"Loaded config from {training_config_path}")
    else:
        logger.warning(f"Config not found at {training_config_path}, using defaults.")
        config = {}

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    infer_config_path = args.infer_config
    if infer_config_path and os.path.exists(infer_config_path):
        infer_config = load_config(infer_config_path)
        logger.info(f"Loaded inference config from {infer_config_path}")
    else:
        logger.warning(
            f"Inference config not found at {infer_config_path}, using defaults."
        )
        infer_config = {}
    infer_cfg = infer_config.get("inference", {})
    filter_cfg = infer_config.get("hotword_filter", {})

    # CLI overrides
    wavlm_id = args.wavlm_model or model_cfg.get("encoder_path", "facebook/wavlm-large")
    llm_id = args.llm_model or model_cfg.get("llm_model_path", "lmsys/vicuna-7b-v1.5")
    encoder_type = args.encoder_type or model_cfg.get("encoder_type", "auto")
    language = args.language or infer_cfg.get("language", "en")
    max_new_tokens = args.max_new_tokens or infer_cfg.get("max_new_tokens", 256)
    num_beams = args.num_beams or infer_cfg.get("num_beams", 4)
    logger.info(f"Encoder type: {encoder_type} | Language: {language}")

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
    hotword_filter = None
    if args.infer_type == "filter":
        filter_method = args.filter_method
        method_defaults = FILTER_DEFAULTS[filter_method]

        common_words_path = args.common_words_path or filter_cfg.get(
            "common_words_path", "data/common_words_5k.txt"
        )
        prob_threshold = (
            args.probability_threshold
            if args.probability_threshold is not None
            else filter_cfg.get(
                "probability_threshold", method_defaults["probability_threshold"]
            )
        )
        word_num = (
            args.word_num
            if args.word_num is not None
            else filter_cfg.get("word_num", method_defaults["word_num"])
        )
        n_gram = filter_cfg.get("n_gram", 2)

        HotwordFilter = load_hotword_filter(filter_method)
        filter_kwargs = dict(
            common_words_path=common_words_path,
            n_gram=n_gram,
            word_num=word_num,
            probability_threshold=prob_threshold,
        )
        if filter_method == "ja":
            # Optional Japanese-specific knobs (all have sensible defaults).
            for key in (
                "tokenizer_backend",
                "reading_weight",
                "surface_bonus",
                "include_homophones",
                "max_homophones",
                "per_word_scoring",
            ):
                if key in filter_cfg:
                    filter_kwargs[key] = filter_cfg[key]
        hotword_filter = HotwordFilter(**filter_kwargs)
        logger.info(
            f"HotwordFilter [{filter_method}]: n_gram={n_gram}, word_num={word_num}, "
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
        encoder_type=encoder_type,
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
            language=language,
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
        log_separator()
        logger.info(f"Evaluating: split={split}, infer_type={args.infer_type}")

        dataset = LibriSpeechDataset(
            data_path=str(data_dir), split=split, config=data_cfg
        )
        logger.info(f"Loaded {len(dataset)} samples from {split}")

        # Select collate function based on infer_type
        if args.infer_type == "nobias":
            collate_fn = get_nobias_collate_fn(
                model.encoder,
                model.decoder_processor,
                device=str(device),
                language=language,
            )
        elif args.infer_type == "gt":
            # For Japanese, annotate GT hotwords with their reading in the prompt.
            reading_fn = None
            if language == "ja":
                from src.utils.japanese import to_katakana

                reading_fn = to_katakana
            collate_fn = get_gt_collate_fn(
                model.encoder,
                model.decoder_processor,
                bias_data=bias_data,
                device=str(device),
                language=language,
                reading_fn=reading_fn,
            )
        elif args.infer_type == "filter":
            collate_fn = get_filter_collate_fn(
                model.encoder,
                model.decoder_processor,
                bias_data=bias_data,
                ctc_data=ctc_data,
                hotword_filter=hotword_filter,
                device=str(device),
                language=language,
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
    log_separator()
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
