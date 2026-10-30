from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

LLAMA3_EOT_TOKEN = "<|eot_id|>"

# Chat scaffolds per LLM family. ``{instruction}`` is the (localised) user
# turn, ``{answer}`` the transcription followed by the end-of-turn token.
#   vicuna: Vicuna v1.5 (LLaMA-2 based), matching the reference recipe.
#   llama3: Llama-3 / 3.1 / 3.2 / 3.3 and derivatives (Swallow, ELYZA-JP, ...).
#           BOS (<|begin_of_text|>) is added by the tokenizer, as <s> is for
#           Vicuna, so it is not part of the template.
PROMPT_FORMATS = {
    "vicuna": {
        "template": "USER: {instruction}\n ASSISTANT:{answer}",
        "eos": "</s>",
    },
    "llama3": {
        "template": (
            "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{answer}"
        ),
        "eos": LLAMA3_EOT_TOKEN,
    },
}


def detect_prompt_format(tokenizer) -> str:
    """Pick the chat scaffold matching the LLM's tokenizer.

    Llama-3 family tokenizers carry the ``<|start_header_id|>`` special token;
    everything else falls back to the Vicuna format.
    """
    tok_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    if tok_id is not None and tok_id != tokenizer.unk_token_id:
        return "llama3"
    return "vicuna"


def _format_hotwords(
    hotwords: List[str],
    readings: Optional[List[str]],
    separator: str,
) -> str:
    """Join hotwords, annotating each with its reading as ``surface（reading）``
    when a parallel *readings* list is supplied (used for Japanese to help the
    LLM disambiguate homophones)."""
    if readings and len(readings) == len(hotwords):
        parts = [
            f"{hw}（{rd}）" if rd and rd != hw else hw
            for hw, rd in zip(hotwords, readings)
        ]
    else:
        parts = list(hotwords)
    return separator.join(parts)


def _instruction(
    filtered_hotwords: List[str],
    language: str,
    readings: Optional[List[str]],
) -> str:
    if language == "ja":
        if not filtered_hotwords:
            return "音声を文字に書き起こしてください。"
        hotwords_str = _format_hotwords(filtered_hotwords, readings, "、")
        return (
            "音声を文字に書き起こしてください。"
            f"次のキーワードが役立つかもしれません：「{hotwords_str}」。"
        )

    # Default: English (matching reference)
    if not filtered_hotwords:
        return "Transcribe speech to text."
    hotwords_str = _format_hotwords(filtered_hotwords, readings, " ")
    return (
        "Transcribe speech to text. Some hotwords might help. "
        f'The hotwords are "{hotwords_str}".'
    )


def create_prompt_with_hotwords(
    filtered_hotwords: List[str],
    transcription: str = "",
    language: str = "en",
    readings: Optional[List[str]] = None,
    prompt_format: str = "vicuna",
) -> str:
    """
    Create prompt text with filtered hotwords for the LLM.

    ``prompt_format="vicuna"`` matches the reference chat format
    ``"USER: {prompt}\\n ASSISTANT:{answer}</s>"``; ``"llama3"`` uses the
    Llama-3 header/``<|eot_id|>`` scaffold (see :func:`detect_prompt_format`).
    The instruction is localised per ``language``.

    Args:
        filtered_hotwords: List of filtered hotwords (surface forms).
        transcription: Transcription text (if provided, the format's
            end-of-turn token is appended).
        language: ``"en"`` (default) or ``"ja"``. Selects the instruction text,
            hotword separator, and reading-annotation behaviour.
        readings: Optional list of readings parallel to ``filtered_hotwords``.
            When given (Japanese), each hotword is annotated as ``橋（ハシ）`` so
            the model can tell same-reading/different-kanji homophones apart.
        prompt_format: ``"vicuna"`` (default) or ``"llama3"``.

    Returns:
        prompt: Prompt text in the selected chat template format.
    """
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(
            f"Unknown prompt_format: {prompt_format}. "
            f"Choose from {', '.join(PROMPT_FORMATS)}."
        )
    fmt = PROMPT_FORMATS[prompt_format]

    # Build answer part with end-of-turn token
    answer = ""
    if transcription:
        answer = f"{transcription}{fmt['eos']}"

    return fmt["template"].format(
        instruction=_instruction(filtered_hotwords, language, readings),
        answer=answer,
    )
