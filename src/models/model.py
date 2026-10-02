import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig, get_peft_model, TaskType
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.projector import AudioProjector
from src.models.encoder import SpeechEncoder
from src.utils.create_prompt import LLAMA3_EOT_TOKEN, detect_prompt_format
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
        encoder_type: str = "auto",  # "auto" | "wavlm"/"hubert"/"wav2vec2" | "whisper"
    ):
        super().__init__()

        logger.info("Initializing CTC-Assisted LLM-Based Contextual ASR Model...")

        # 1. Load speech encoder (WavLM/HuBERT-CTC or Whisper) behind a common
        #    interface, so the rest of the model is encoder-agnostic.
        try:
            self.encoder = SpeechEncoder(
                encoder_id,
                encoder_type=encoder_type,
                freeze=freeze_encoder,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            # Backward-compatible aliases used elsewhere in the codebase.
            self.encoder_processor = self.encoder.processor
            self.encoder_hidden_size = self.encoder.hidden_size
        except Exception as e:
            logger.error(f"Error loading encoder {encoder_id}: {e}")
            raise
        self.conv_stride = 5  # Projector downsampling factor (frames per token)

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

        # Chat scaffold for the prompt ("vicuna" or "llama3"), and the token ids
        # that end the assistant turn during generation. Llama-3 answers are
        # terminated with <|eot_id|>, which differs from eos_token on base models.
        self.prompt_format = detect_prompt_format(self.decoder_processor)
        self.stop_token_ids = [self.decoder_processor.eos_token_id]
        if self.prompt_format == "llama3":
            eot_id = self.decoder_processor.convert_tokens_to_ids(LLAMA3_EOT_TOKEN)
            if eot_id not in self.stop_token_ids:
                self.stop_token_ids.append(eot_id)
        logger.info(
            f"Prompt format: {self.prompt_format} | Stop token ids: {self.stop_token_ids}"
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

        # 5. Gradient checkpointing (encoder GC is handled inside SpeechEncoder)
        if use_gradient_checkpointing:
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
    ) -> torch.Tensor:
        """Extract encoder hidden states.

        ``audio_input`` is the encoder's native input: a 2D waveform batch for
        WavLM/HuBERT-CTC encoders, or a 3D log-mel batch for Whisper. The
        :class:`SpeechEncoder` wrapper handles the difference.
        """
        try:
            return self.encoder.extract_features(audio_input, attention_mask)
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
        encoder_lengths = self.encoder.output_lengths(encoder_attention_mask)
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
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.decoder_processor.pad_token_id,
            **kwargs,
        )
        return generated_ids
