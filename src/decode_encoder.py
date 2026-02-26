import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    WavLMForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)
from jiwer import wer as compute_wer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataset import LibriSpeechDataset

# Logger
logger: Optional[logging.Logger] = None


def setup_logger(output_dir: str):
    """Setup logging."""
    global logger
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/inference.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def remove_special_characters(text: str) -> str:
    """Clean text."""
    import re

    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
    return re.sub(chars_to_ignore_regex, "", text).lower()


class InferenceDataCollator:
    """Data collator for inference with variable-length inputs."""

    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict]) -> Dict:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Keep metadata
        batch["utt_ids"] = [feature["utt_id"] for feature in features]
        batch["transcriptions"] = [feature["transcription"] for feature in features]

        return batch


class LazyPreprocessedDataset(torch.utils.data.Dataset):
    """Lazy preprocessing dataset for inference."""

    def __init__(self, base_dataset, processor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        # Load and process audio
        audio = sample["audio"].squeeze(0).numpy()
        inputs = self.processor(
            audio,
            sampling_rate=sample["sampling_rate"],
        )

        return {
            "input_values": inputs.input_values[0],
            "transcription": sample["transcription"],
            "utt_id": sample["utterance_id"],
            "speaker_id": sample["speaker_id"],
        }


def load_model_and_processor(model_path: str, vocab_path: str, device: str = "cuda"):
    """Load fine-tuned model and processor."""

    logger.info(f"Loading model from {model_path}")

    # Initialize tokenizer with vocab
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

    # Load model
    model = WavLMForCTC.from_pretrained(
        model_path,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    model.eval()
    logger.info(f"Model loaded successfully on {device}")

    return model, processor, tokenizer


@torch.no_grad()
def run_inference(
    model: WavLMForCTC,
    processor: Wav2Vec2Processor,
    data_loader: DataLoader,
    device: str = "cuda",
):
    """Run inference on test set."""
    results = []
    pred_texts = []
    true_texts = []

    logger.info("Running inference...")
    for batch in tqdm(data_loader, desc="Inference"):
        # Move batch to device
        input_values = batch["input_values"].to(device)

        # Forward pass
        outputs = model(input_values)
        logits = outputs.logits

        # Get predictions
        pred_ids = torch.argmax(logits, dim=-1)

        # Decode predictions
        pred_str = processor.batch_decode(pred_ids)

        # Process true texts
        label_str = batch["transcriptions"]

        # Store results
        for utt_id, pred, true in zip(batch["utt_ids"], pred_str, label_str):
            pred_cleaned = remove_special_characters(pred)
            true_cleaned = remove_special_characters(true)
            results.append(
                {
                    "utt_id": utt_id,
                    "prediction": pred_cleaned,
                    "reference": true_cleaned,
                    "wer": compute_wer(true_cleaned, pred_cleaned),
                }
            )
            pred_texts.append(pred_cleaned)
            true_texts.append(true_cleaned)

    return results, pred_texts, true_texts


def compute_metrics(pred_texts: List[str], true_texts: List[str]) -> Dict:
    """Compute WER and other metrics."""

    # Clean references like in training
    true_texts_cleaned = [remove_special_characters(text) for text in true_texts]
    pred_texts_cleaned = [remove_special_characters(text) for text in pred_texts]

    wer = compute_wer(true_texts_cleaned, pred_texts_cleaned)

    logger.info(f"Word Error Rate (WER): {wer:.4f}")

    return {"wer": wer, "num_samples": len(pred_texts)}


def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: str,
):
    """Save inference results and metrics."""

    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(output_dir, "inference_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    # Save predictions in a simple format
    predictions_file = os.path.join(output_dir, "predictions.txt")
    with open(predictions_file, "w") as f:
        for result in results:
            f.write(f"{result['utt_id']}\n")
            f.write(f"  Prediction: {result['prediction']}\n")
            f.write(f"  Reference:  {result['reference']}\n")
            f.write(f"  WER: {result['wer']:.4f}\n")
            f.write("\n")
    logger.info(f"Predictions saved to {predictions_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on test set with fine-tuned WavLM model"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech",
        help="Path to LibriSpeech dataset directory",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="exp/wavlm-ft-libri/final",
        help="Path to fine-tuned WavLM model",
    )

    parser.add_argument(
        "--vocab_path",
        type=str,
        default="exp/wavlm-ft-libri/vocab.json",
        help="Path to vocabulary file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp/wavlm-ft-libri/inference_results",
        help="Path to save inference results",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_arguments()

    # Setup logger
    logger = setup_logger(args.output_dir)

    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    # Check if vocab files exist
    if not os.path.exists(args.vocab_path):
        logger.error(f"Vocab path does not exist: {args.vocab_path}")
        return

    # Load model and processor
    model, processor, tokenizer = load_model_and_processor(
        args.model_path, args.vocab_path, args.device
    )

    # Process each test set separately
    test_sets = [
        # ("test-clean", f"{args.output_dir}/test-clean"),
        # ("test-other", f"{args.output_dir}/test-other"),
        ("dev-clean", f"{args.output_dir}/dev-clean"),
        ("dev-other", f"{args.output_dir}/dev-other"),
    ]

    all_metrics = {}
    import time

    start_time = time.time()
    for split_name, split_output_dir in test_sets:
        logger.info(f"Processing {split_name} split")
        logger.info("=" * 80)

        # Load test dataset for this split
        logger.info(f"Loading {split_name} dataset...")
        test_dataset = LibriSpeechDataset(data_path=args.data_dir, split=split_name)
        logger.info(f"Total {split_name} samples: {len(test_dataset)}")

        # Preprocess dataset
        logger.info("Preprocessing dataset...")
        test_dataset = LazyPreprocessedDataset(test_dataset, processor)

        # Create data loader
        data_collator = InferenceDataCollator(processor, padding=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Run inference
        results, pred_texts, true_texts = run_inference(
            model, processor, test_loader, args.device
        )

        # Compute metrics
        metrics = compute_metrics(pred_texts, true_texts)
        all_metrics[split_name] = metrics

        # Save results
        save_results(results, metrics, split_output_dir)

        logger.info("=" * 80)
        logger.info(f"{split_name} completed!")
        logger.info(f"Total samples processed: {len(results)}")
        logger.info(f"Word Error Rate (WER): {metrics['wer']:.4f}")
        logger.info("=" * 80)

    end_time = time.time()
    elapsed_time = end_time - start_time
    # Save combined metrics
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Total inference time: {elapsed_time:.2f} seconds - {elapsed_time/60:.2f} minutes - {elapsed_time/3600:.2f} hours"
    )
    combined_metrics_file = os.path.join(args.output_dir, "all_metrics.json")
    with open(combined_metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Combined metrics saved to {combined_metrics_file}")

    for split, metrics in all_metrics.items():
        logger.info(f"{split}: WER = {metrics['wer']:.4f}")


if __name__ == "__main__":
    main()
