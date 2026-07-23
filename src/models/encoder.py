"""Unified speech-encoder wrapper for the contextual ASR model.

The projector + LLM stack only needs three things from an audio encoder:

* a way to turn raw waveforms into model inputs (:meth:`process_audio`),
* a way to turn those inputs into a sequence of hidden states
  (:meth:`extract_features`), and
* the number of *valid* output frames for a given input, so padding can be
  masked out (:meth:`output_lengths`).

This wrapper implements that small interface for two very different encoder
families behind a single ``encoder_type`` switch:

* ``wavlm`` / ``hubert`` / ``wav2vec2`` — self-supervised CTC-style encoders that
  consume a raw waveform (``input_values``) and expose
  ``_get_feat_extract_output_lengths``.
* ``whisper`` — a seq2seq encoder that consumes an 80/128-bin log-mel
  spectrogram (``input_features``), always padded to 30 s, and downsamples the
  mel frames by 2 via its second conv layer.

By funnelling both through the same ``{"input_values", "encoder_attention_mask"}``
batch schema (for Whisper, ``input_values`` carries the mel features), the rest
of the code — collators, ``model.embed`` — stays encoder-agnostic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCTC,
    AutoProcessor,
)

logger = logging.getLogger(__name__)

# HF ``model_type`` values handled as raw-waveform CTC encoders.
_CTC_MODEL_TYPES = {
    "wavlm",
    "hubert",
    "wav2vec2",
    "wav2vec2-conformer",
    "unispeech",
    "sew",
    "sew-d",
}
_WHISPER_MODEL_TYPES = {"whisper"}


def _load_feature_extractor(encoder_id: str):
    """Load an audio feature extractor, tolerating checkpoints that only ship a
    processor (finetuned CTC dirs) or only a feature extractor (base models)."""
    try:
        proc = AutoProcessor.from_pretrained(encoder_id)
        # A processor may bundle a feature extractor + tokenizer; we only need
        # the audio side for feature extraction.
        fe = getattr(proc, "feature_extractor", None)
        return proc, (fe if fe is not None else proc)
    except Exception as exc:  # pragma: no cover - depends on checkpoint
        logger.warning(
            "AutoProcessor failed for %s (%s); falling back to AutoFeatureExtractor",
            encoder_id,
            exc,
        )
        fe = AutoFeatureExtractor.from_pretrained(encoder_id)
        return fe, fe


class SpeechEncoder(nn.Module):
    """Wrap a WavLM/HuBERT-CTC or Whisper encoder behind a common interface.

    Args:
        encoder_id: HF hub id or local path of the encoder checkpoint.
        encoder_type: ``"auto"`` (default), ``"wavlm"``/``"hubert"``/``"wav2vec2"``,
            or ``"whisper"``. ``"auto"`` inspects the model config.
        freeze: Freeze all encoder parameters (default: True).
        use_gradient_checkpointing: Enable gradient checkpointing on the encoder.
    """

    def __init__(
        self,
        encoder_id: str,
        encoder_type: str = "auto",
        freeze: bool = True,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_id = encoder_id
        self.encoder_type = self._resolve_type(encoder_id, encoder_type)
        logger.info(
            "Loading %s encoder: %s (type=%s)",
            self.encoder_type,
            encoder_id,
            encoder_type,
        )

        if self.encoder_type == "whisper":
            self._init_whisper(encoder_id)
        else:
            self._init_ctc(encoder_id)

        if freeze:
            self.model.requires_grad_(False)
            logger.info("%s encoder frozen", self.encoder_type)

        if use_gradient_checkpointing:
            self.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_type(encoder_id: str, encoder_type: str) -> str:
        if encoder_type and encoder_type != "auto":
            return encoder_type.lower()
        # Cheap name-based hint first (avoids a config download for the obvious
        # case), then fall back to the model config's ``model_type``.
        if "whisper" in encoder_id.lower():
            return "whisper"
        try:
            cfg = AutoConfig.from_pretrained(encoder_id)
            model_type = getattr(cfg, "model_type", "").lower()
            if model_type in _WHISPER_MODEL_TYPES:
                return "whisper"
            if model_type in _CTC_MODEL_TYPES:
                return model_type
        except Exception as exc:  # pragma: no cover - depends on checkpoint
            logger.warning("Could not read config for %s: %s", encoder_id, exc)
        # Default to the CTC/raw-waveform path.
        return "wavlm"

    def _init_whisper(self, encoder_id: str) -> None:
        from transformers import WhisperModel

        self.processor, self.feature_extractor = _load_feature_extractor(encoder_id)
        whisper = WhisperModel.from_pretrained(encoder_id)
        # We only need the encoder tower.
        self.model = whisper.get_encoder()
        self.hidden_size = whisper.config.d_model
        self.input_key = "input_features"
        # Whisper's second conv layer has stride 2 (mel frames -> output frames).
        self._max_output_frames = whisper.config.max_source_positions  # 1500

    def _init_ctc(self, encoder_id: str) -> None:
        self.processor, self.feature_extractor = _load_feature_extractor(encoder_id)
        try:
            self.model = AutoModelForCTC.from_pretrained(encoder_id)
        except Exception as exc:
            logger.warning(
                "AutoModelForCTC failed for %s (%s); using AutoModel", encoder_id, exc
            )
            self.model = AutoModel.from_pretrained(encoder_id)
        self.hidden_size = self.model.config.hidden_size
        self.input_key = "input_values"
        self._max_output_frames = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info(
                "Gradient checkpointing enabled on %s encoder", self.encoder_type
            )

    def process_audio(
        self,
        audio_list: List,
        sampling_rate: int = 16000,
        device: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Turn a list of waveforms into a batch dict.

        Returns ``{"input_values": <encoder input>, "encoder_attention_mask":
        <validity mask>}``. For Whisper ``input_values`` holds the log-mel
        features and the mask spans mel frames; for CTC encoders it holds the
        (normalised) waveform and the mask spans samples.
        """
        if self.encoder_type == "whisper":
            feats = self.feature_extractor(
                audio_list,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = feats["input_features"]
            mask = feats.get("attention_mask", None)
            if mask is None:
                mask = torch.ones(inputs.shape[0], inputs.shape[-1], dtype=torch.long)
        else:
            feats = self.feature_extractor(
                audio_list,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
            )
            inputs = feats["input_values"]
            mask = feats["attention_mask"]

        out = {"input_values": inputs, "encoder_attention_mask": mask}
        if device:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    def extract_features(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the last hidden state ``(batch, frames, hidden_size)``."""
        if self.encoder_type == "whisper":
            # The Whisper encoder ignores the attention mask (fixed 30 s window).
            output = self.model(inputs)
            return output.last_hidden_state
        output = self.model(
            inputs, attention_mask=attention_mask, output_hidden_states=True
        )
        return output.hidden_states[-1]

    def output_lengths(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Valid encoder-output frame count for each item in the batch."""
        input_lengths = attention_mask.sum(dim=1)
        if self.encoder_type == "whisper":
            # Two conv layers; the second has stride 2, kernel 3, padding 1:
            #   out = floor((L - 1) / 2) + 1
            lengths = torch.div(input_lengths - 1, 2, rounding_mode="floor") + 1
            lengths = lengths.clamp(min=0, max=self._max_output_frames)
            return lengths
        return self.model._get_feat_extract_output_lengths(input_lengths)
