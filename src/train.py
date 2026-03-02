import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from whisper.normalizers import BasicTextNormalizer
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import CTCAssistedLLMASR
from src.utils import (
    load_config,
    save_config,
    save_info,
    save_checkpoint,
    load_checkpoint,
    compute_wer,
    setup_logging,
    switch_log_file,
    set_seed,
)
from src.hotword_filter import create_prompt_with_hotwords
from src.dataset import KaldiDataset, LibriSpeechDataset

logger = logging.getLogger(__name__)


def get_collate_fn(
    encoder_processor,
    decoder_processor,
    mode: str = "train",
    device: str = "cuda",
):
    def collate_fn_cpu(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate: pad audio + tạo attention mask + tokenize transcriptions"""
        encoder_inputs = encoder_processor(
            [sample["audio"] for sample in batch],
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        # Calculate prompt lengths for masking
        prompts_only = [
            create_prompt_with_hotwords(sample["hotwords"]) for sample in batch
        ]
        prompt_only_inputs = decoder_processor(
            prompts_only,
            return_tensors="pt",
            padding=True,
        )
        prompt_lengths = (prompt_only_inputs.attention_mask == 1).sum(dim=1).tolist()

        # Full prompt + transcription (for train loss / eval loss)
        full_prompts = [
            create_prompt_with_hotwords(sample["hotwords"], sample["transcription"])
            for sample in batch
        ]
        full_decoder_inputs = decoder_processor(
            full_prompts,
            return_tensors="pt",
            padding=True,
        )

        if mode == "train":
            return {
                "input_values": encoder_inputs.input_values,
                "input_ids": full_decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs.attention_mask,
                "decoder_attention_mask": full_decoder_inputs.attention_mask,
                "prompt_lengths": prompt_lengths,
            }
        elif mode == "eval":
            return {
                "input_values": encoder_inputs.input_values,
                # Full prompt+transcription for loss computation
                "input_ids": full_decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs.attention_mask,
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

    def collate_fn_gpu(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoder_inputs = encoder_processor(
            [sample["audio"] for sample in batch],
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        ).to("cuda")

        prompts_only = [
            create_prompt_with_hotwords(sample["hotwords"]) for sample in batch
        ]
        prompt_only_inputs = decoder_processor(
            prompts_only,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        prompt_lengths = (prompt_only_inputs.attention_mask == 1).sum(dim=1).tolist()

        # Full prompt + transcription (for train loss / eval loss)
        full_prompts = [
            create_prompt_with_hotwords(sample["hotwords"], sample["transcription"])
            for sample in batch
        ]
        full_decoder_inputs = decoder_processor(
            full_prompts,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        if mode == "train":
            return {
                "input_values": encoder_inputs.input_values,
                "input_ids": full_decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs.attention_mask,
                "decoder_attention_mask": full_decoder_inputs.attention_mask,
                "prompt_lengths": prompt_lengths,
            }
        elif mode == "eval":
            return {
                "input_values": encoder_inputs.input_values,
                # Full prompt+transcription for loss computation
                "input_ids": full_decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs.attention_mask,
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


class Trainer:
    def __init__(
        self,
        model: CTCAssistedLLMASR,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        self.device = torch.device("cuda")
        self.model.to(self.device)

        # Step-based config
        self.max_steps = config["max_steps"]  # tổng optimizer steps
        self.eval_every_steps = config.get("eval_every", 5000)
        self.save_every_steps = config.get("save_every", 10000)
        self.log_every_steps = config.get("log_every", 100)

        # DataLoaders
        self.num_workers = config.get("num_workers", 0)
        self.data_loading_device = "cuda" if self.num_workers == 0 else "cpu"
        if config.get("pin_memory", None) is not None:
            self.pin_memory = config["pin_memory"]
        else:
            self.pin_memory = self.data_loading_device == "cpu"
        logger.warning(
            f"Data loading device: {self.data_loading_device}"
            f"(num_workers={self.num_workers}-pin_memory={self.pin_memory})"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=get_collate_fn(
                self.model.encoder_processor,
                self.model.decoder_processor,
                mode="train",
                device=self.data_loading_device,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
        )
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=get_collate_fn(
                    self.model.encoder_processor,
                    self.model.decoder_processor,
                    mode="eval",
                    device=self.data_loading_device,
                ),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                prefetch_factor=2,
            )
            self.normalizer = BasicTextNormalizer()

        # Optimizer – chỉ train projector AND LoRA weights
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=float(config["learning_rate"]),
            betas=(config["beta1"], config["beta2"]),
            weight_decay=config["weight_decay"],
        )

        # Scheduler (theo max_steps)
        self.scheduler = get_linear_schedule(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=self.max_steps,
        )

        self.grad_accumulation = config.get("grad_accumulation", 1)
        # Note: GradScaler not needed with bfloat16 native amp
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)
        self.save_total_limit = config.get("save_total_limit", None)

        # TensorBoard Writer with timestamped run directory
        if output_dir:
            self.writer = SummaryWriter(tensorboard_log_dir(output_dir))
        else:
            self.writer = None

        # State
        self.global_step = 0
        self.best_loss = float("inf")

        if config.get("checkpoint_dir") is None:
            self.checkpoint_dir = None
            logger.warning(
                "No checkpoint_dir specified in config. Checkpoints will not be saved."
            )
        else:
            self.checkpoint_dir = Path(config["checkpoint_dir"])
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log files
        self.output_dir = output_dir
        self.train_log_file = (
            str(Path(output_dir) / "logs" / "training.log") if output_dir else None
        )
        self.eval_log_file = (
            str(Path(output_dir) / "logs" / "eval.log") if output_dir else None
        )

        logger.info(f"Trainer initialized")
        logger.info(f"Max optimizer steps: {self.max_steps}")
        logger.info(f"Trainable parameters: {model.count_trainable_parameters():,}")

        # Calculate steps per epoch for epoch tracking
        self.steps_per_epoch = len(self.train_loader) // self.grad_accumulation
        logger.info(f"Steps per epoch: {self.steps_per_epoch}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        try:
            _, step, loss = load_checkpoint(
                self.model,
                checkpoint_path,
                optimizer=self.optimizer,
                device=str(self.device),
            )
            self.global_step = step
            self.best_loss = loss
            # Fast-forward scheduler to current step
            self.scheduler.last_epoch = step
            logger.info(f"Resumed from step {step}, best_loss={loss:.4f}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints to keep only save_total_limit most recent ones.

        Keeps best_model.pt and the N most recent step checkpoints,
        where N = self.save_total_limit.
        """
        if self.save_total_limit is None or self.checkpoint_dir is None:
            return

        import re

        # Collect step-based checkpoints (exclude best_model.pt)
        step_checkpoints = []
        for ckpt_file in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            match = re.search(r"checkpoint_step_(\d+)\.pt$", ckpt_file.name)
            if match:
                step_checkpoints.append((int(match.group(1)), ckpt_file))

        # Sort by step number (ascending)
        step_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest checkpoints beyond the limit
        while len(step_checkpoints) > self.save_total_limit:
            _, oldest_ckpt = step_checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {oldest_ckpt}")
            oldest_ckpt.unlink()

    def to_cuda(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Đẩy tất cả Tensor trong batch lên device, giữ nguyên các kiểu dữ liệu khác."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def evaluate(
        self,
        max_new_tokens: int = 200,
        num_beams: int = 4,
        do_sample: bool = False,
    ) -> tuple:
        # Switch to eval log file
        if self.eval_log_file:
            switch_log_file(self.eval_log_file)

        self.model.eval()
        total_loss = 0
        num_batches = 0
        avg_loss = 0.0

        hyps = []
        refs = []
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            if self.data_loading_device == "cpu":
                batch = self.to_cuda(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = self.model(**batch)
            total_loss += loss.item()
            num_batches += 1

            generated_ids = self.model.generate(
                input_values=batch["input_values"],
                input_ids=batch["generate_input_ids"],
                encoder_attention_mask=batch["encoder_attention_mask"],
                decoder_attention_mask=batch["generate_attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

            predicted_texts = self.model.decoder_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            hyps.extend(predicted_texts)
            refs.extend(batch["transcription"])
            logger.info(
                f"Batch {num_batches}\n"
                f"Predicted: {predicted_texts}\n"
                f"Refs: {batch['transcription']}"
            )

        hyps = [self.normalizer(hyp) for hyp in hyps]
        refs = [
            (
                self.normalizer(ref)
                if isinstance(ref, str)
                else [self.normalizer(ref_i) for ref_i in ref]
            )
            for ref in refs
        ]

        avg_loss = total_loss / max(num_batches, 1)
        wer = compute_wer(hyps, refs) if hyps and refs else 0.0
        logger.info(
            f"Step {self.global_step} | Eval Loss: {avg_loss:.4f} | WER: {wer:.4f}"
        )

        if self.writer:
            self.writer.add_scalar("eval/loss", avg_loss, self.global_step)
            self.writer.add_scalar("eval/wer", wer, self.global_step)

        # Switch back to training log file
        if self.train_log_file:
            switch_log_file(self.train_log_file)

        self.model.train()
        return avg_loss, wer

    def train(self) -> None:
        logger.info(
            f"Starting training from step {self.global_step} to {self.max_steps}..."
        )
        self.model.cuda()

        progress_bar = tqdm(
            initial=self.global_step,
            total=self.max_steps,
            desc="Training",
        )

        try:
            while self.global_step < self.max_steps:
                self.model.train()
                self.model.encoder.eval()
                self.model.llm.eval()
                for batch_idx, batch in enumerate(self.train_loader):
                    if self.data_loading_device == "cpu":
                        batch = self.to_cuda(batch)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = self.model(**batch)
                        loss = loss / self.grad_accumulation
                    loss.backward()

                    if (batch_idx + 1) % self.grad_accumulation == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.projector.parameters(), self.clip_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        self.global_step += 1

                        # Logging
                        if self.global_step % self.log_every_steps == 0 and self.writer:
                            lr = self.optimizer.param_groups[0]["lr"]
                            actual_loss = loss.item() * self.grad_accumulation
                            progress_bar.set_postfix(
                                loss=f"{actual_loss:.4f}",
                                lr=f"{lr:.2e}",
                                grad_norm=f"{grad_norm:.2f}",
                            )
                            self.writer.add_scalar(
                                "train/loss", actual_loss, self.global_step
                            )
                            self.writer.add_scalar(
                                "train/learning_rate", lr, self.global_step
                            )
                            self.writer.add_scalar(
                                "train/grad_norm",
                                grad_norm.item(),
                                self.global_step,
                            )
                            self.writer.add_scalar(
                                "train/epoch",
                                self.global_step / self.steps_per_epoch,
                                self.global_step,
                            )

                        progress_bar.update(1)

                        # Save checkpoint
                        if self.global_step % self.save_every_steps == 0:
                            ckpt_path = (
                                self.checkpoint_dir
                                / f"checkpoint_step_{self.global_step}.pt"
                            )
                            save_checkpoint(
                                self.model,
                                self.optimizer,
                                0,
                                self.global_step,
                                loss.item(),
                                str(ckpt_path),
                            )
                            logger.info(f"Saved checkpoint at step {self.global_step}")
                            self._cleanup_checkpoints()

                        # Evaluate
                        if (
                            self.val_dataset
                            and self.global_step % self.eval_every_steps == 0
                        ):
                            logger.info(f"Evaluating at step {self.global_step}...")
                            val_loss, wer = self.evaluate()
                            logger.info(
                                f"Step {self.global_step} | Eval Loss: {val_loss:.4f} | WER: {wer:.4f}"
                            )
                            if val_loss < self.best_loss:
                                self.best_loss = val_loss
                                best_path = self.checkpoint_dir / "best_model.pt"
                                save_checkpoint(
                                    self.model,
                                    self.optimizer,
                                    0,
                                    self.global_step,
                                    val_loss,
                                    str(best_path),
                                    is_best=True,
                                )
                                logger.info(
                                    f"New best model at step {self.global_step}"
                                )
                        self.model.train()
                        self.model.encoder.eval()
                        self.model.llm.eval()

                    # Break if max steps reached
                    if self.global_step >= self.max_steps:
                        break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        except RuntimeError as e:
            logger.error(f"Runtime error during training: {e}")
            raise
        finally:
            progress_bar.close()
            if self.writer:
                self.writer.close()
        logger.info("Training completed!")

    def train_epoch(self, num_epochs: int = 1, output_dir: str = "exp") -> None:
        """Train for one epoch (not used in current step-based training)"""
        self.model.train()
        step = 0
        model_dir = Path(output_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            self.model.encoder.eval()
            self.model.llm.eval()
            for batch_idx, batch in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch}")
            ):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.model(**batch)
                    loss = loss / self.grad_accumulation
                loss.backward()

                if (batch_idx + 1) % self.grad_accumulation == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    step += 1

                    if self.writer:
                        lr = self.optimizer.param_groups[0]["lr"]
                        self.writer.add_scalar("train/loss", loss.item(), step)
                        self.writer.add_scalar("train/learning_rate", lr, step)
                        self.writer.add_scalar(
                            "train/grad_norm",
                            grad_norm.item(),
                            step,
                        )
            self.evaluate()
            self.model.save_pretrained(str(Path(output_dir) / "models"))


def main():
    set_seed(42)
    import argparse

    parser = argparse.ArgumentParser(description="Train CTC-assisted LLM ASR")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="exp")
    parser.add_argument("--load-checkpoint", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=str(f"{output_dir}/logs/training.log"), level=logging.INFO)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    full_config = load_config(str(config_path))

    # Data
    logger.info(f"Loading datasets from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    data_config = full_config.get("data", {})
    train_splits = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]
    logger.info(f"Train splits: {', '.join(train_splits)}")
    train_dataset = ConcatDataset(
        [
            LibriSpeechDataset(data_path=data_dir, split=s, config=data_config)
            for s in train_splits
        ]
    )

    val_splits = [
        "dev-clean",
        # "dev-other",
    ]
    logger.info(f"Val splits: {', '.join(val_splits)}")
    val_dataset = ConcatDataset(
        [
            LibriSpeechDataset(data_path=data_dir, split=s, config=data_config)
            for s in val_splits
        ]
    )

    # Model
    model_config = full_config["model"]
    model = CTCAssistedLLMASR(
        encoder_id=model_config["wavlm_model_path"],
        decoder_id=model_config["llm_model_path"],
        freeze_encoder=model_config["freeze_encoder"],
        freeze_llm=model_config["freeze_llm"],
        use_lora=model_config.get("use_lora", False),
    )

    train_config = full_config["training"]
    train_config["checkpoint_dir"] = str(output_dir / "checkpoints")
    # Trainer
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=args.output_dir,
    )

    # Resume
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        trainer._load_checkpoint(args.load_checkpoint)

    # Save config
    save_config(full_config, str(output_dir / "training_config.yaml"))

    # Train
    if train_config.get("num_epochs", 0) > 0:
        logger.info(f"Starting training for {train_config['num_epochs']} epochs")
        trainer.train_epoch(
            num_epochs=train_config["num_epochs"],
            output_dir=args.output_dir,
        )
    else:
        logger.info(f"Starting training for {train_config['max_steps']} steps")
        trainer.train()
    logger.info("Training completed successfully!")

    save_info(
        {
            "status": "completed",
            "max_steps": train_config["max_steps"],
            "best_loss": float(trainer.best_loss),
            "results_dir": str(output_dir),
        },
        str(output_dir / "training_info.yaml"),
    )


if __name__ == "__main__":
    import time

    start_time = time.time()

    main()

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Total time: {elapsed/3600:.2f} hours - {elapsed/(3600*24):.2f} days")
