import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    WavLMForCTC,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
)

from peft import LoraConfig, get_peft_model, TaskType
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.projector import AudioProjector
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CTCAssistedLLMASR(nn.Module):
    def __init__(
        self,
        encoder_id: str = "facebook/wavlm-large",
        decoder_id: str = "lmsys/vicuna-7b-v1.5",
        freeze_encoder: bool = True,
        freeze_llm: bool = True,
        use_lora: bool = False,  # Added LoRA flag
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        logger.info("Initializing CTC-Assisted LLM-Based Contextual ASR Model...")

        # 1. Load WavLM Encoder
        try:
            logger.info(f"Loading WavLM encoder: {encoder_id}")
            self.encoder = WavLMForCTC.from_pretrained(encoder_id)
            self.encoder_processor = AutoProcessor.from_pretrained(encoder_id)
            self.encoder_hidden_size = self.encoder.config.hidden_size
        except Exception as e:
            logger.error(f"Error loading WavLM: {e}")
            raise
        self.conv_stride = 5  # Store for attention mask calculation

        if freeze_encoder:
            self.encoder.requires_grad_(False)
            logger.info("WavLM encoder frozen")

        # 3. Load LLM
        logger.info(f"Loading LLM: {decoder_id}")
        self.decoder_processor = AutoTokenizer.from_pretrained(
            decoder_id, use_fast=True
        )

        if self.decoder_processor.pad_token is None:
            self.decoder_processor.pad_token = self.decoder_processor.eos_token
        logger.info(
            f"Tokenizer loaded. Pad token: {self.decoder_processor.pad_token}, "
            f"EOS token: {self.decoder_processor.eos_token}"
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            decoder_id, torch_dtype=torch.bfloat16
        )

        self.llm_hidden_size = self.llm.config.hidden_size

        if freeze_llm:
            self.llm.requires_grad_(False)
            logger.info("LLM frozen")

            if use_lora:
                logger.info("Applying LoRA to the frozen LLM...")
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=[
                        "q_proj",
                        "v_proj",
                    ],  # Standard for LLaMA/Vicuna architectures
                )
                self.llm = get_peft_model(self.llm, lora_config)
                self.llm.print_trainable_parameters()
                logger.info("LoRA applied to LLM")

        # 4. Audio Projector (matching reference: ReLU, no dropout, no LayerNorm)
        self.projector = AudioProjector(
            input_dim=self.encoder_hidden_size,
            hidden_dim=2048,
            output_dim=self.llm_hidden_size,
            conv_kernel_size=self.conv_stride,
        )

        # 5. Gradient checkpointing
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            self.llm.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        logger.info("Model initialization completed!")
        logger.info(f"Trainable parameters: {self.count_trainable_parameters():,}")

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_audio(
        self,
        audio_input: torch.FloatTensor,
        attention_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Validate input
        if audio_input.dim() != 2:
            raise ValueError(
                f"audio_input must be 2D (batch, length), got {audio_input.shape}"
            )
        try:
            encoder_output = self.encoder(
                audio_input, attention_mask=attention_mask, output_hidden_states=True
            )
            audio_features = encoder_output.hidden_states[-1]
            return audio_features
        except Exception as e:
            logger.error(f"Error in encode_audio: {e}")
            raise

    def embed(
        self,
        input_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Embedding audio

        Args:
            audio_input (`torch.Tensor` of shape `(batch_size, audio_length)`):
                Input audio waveform.

        Returns:
            audio_embeddings (`torch.Tensor` of shape `(batch_size, downsampled_length, llm_hidden_size)`):
                Projected audio embeddings.
        """
        # 1. Encode audio
        logger.debug(f"Audio input dtype: {input_values.dtype}")
        logger.debug(f"Audio input shape: {input_values.shape}")
        logger.debug(f"Encoder attention mask dtype: {encoder_attention_mask.dtype}")
        audio_features = self.encode_audio(input_values, encoder_attention_mask)

        # 2. Project audio features
        audio_embeds = self.projector(audio_features)

        # 3. Calculate projected audio lengths và mask
        encoder_lengths = self.encoder._get_feat_extract_output_lengths(
            encoder_attention_mask.sum(dim=1)
        )
        proj_lengths = encoder_lengths // self.projector.conv.kernel_size[0]
        max_proj_len = proj_lengths.max().item()
        audio_embeds = audio_embeds[:, :max_proj_len]

        # 4. Get decoder embeddings
        decoder_embeds = self.llm.get_input_embeddings()(input_ids)

        # 5. Concatenate audio and text embeddings
        inputs_embeds = torch.cat([audio_embeds, decoder_embeds], dim=1)

        # 6. Compute combined attention mask
        audio_mask = (
            torch.arange(
                max_proj_len,
                device=decoder_attention_mask.device,
            ).unsqueeze(0)
            < proj_lengths.unsqueeze(1)
        ).long()
        attention_mask = torch.cat([audio_mask, decoder_attention_mask], dim=1)

        return inputs_embeds, attention_mask

    def forward(
        self,
        input_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        prompt_lengths: Optional[List[int]] = None,  # Added prompt_lengths
        **kwargs,
    ):
        """
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_length)`):
                Raw audio features.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Token ids.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, feature_length)`):
                1: non-mask
                0: mask
            decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                1: non-mask
                0: mask
        """
        # 1. Embed audio and prompt
        inputs_embeds, combined_attention_mask = self.embed(
            input_values,
            input_ids,
            encoder_attention_mask,
            decoder_attention_mask,
        )

        # 2. Mask padding and prompts
        labels = input_ids.clone()
        labels[decoder_attention_mask == 0] = -100

        # Mask out the prompt tokens so loss is only calculated on the transcription
        if prompt_lengths is not None:
            for i, length in enumerate(prompt_lengths):
                labels[i, :length] = -100

        # 3. Pad labels with audio part
        labels = F.pad(
            labels, (inputs_embeds.shape[1] - input_ids.shape[1], 0), value=-100
        )

        # 4. Forward through LLM
        decoder_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True,
        )

        return decoder_outputs.loss

    @torch.amp.autocast("cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def generate(
        self,
        input_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        max_new_tokens: int = 1024,
        num_beams: int = 4,
        **kwargs,
    ):
        # 1. Embed audio and prompt
        inputs_embeds, attention_mask = self.embed(
            input_values, input_ids, encoder_attention_mask, decoder_attention_mask
        )

        # 2. Generate
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            eos_token_id=self.decoder_processor.eos_token_id,
            pad_token_id=self.decoder_processor.pad_token_id,
            **kwargs,
        )
        return generated_ids
