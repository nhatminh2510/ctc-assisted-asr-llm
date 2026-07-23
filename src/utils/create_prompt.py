from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


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


def create_prompt_with_hotwords(
    filtered_hotwords: List[str],
    transcription: str = "",
    language: str = "en",
    readings: Optional[List[str]] = None,
) -> str:
    """
    Create prompt text with filtered hotwords for the LLM.

    Matches the reference Vicuna chat format
    ``"USER: {prompt}\\n ASSISTANT:{answer}</s>"``. The instruction is localised
    per ``language`` while the USER/ASSISTANT scaffold is kept stable so the
    rest of the pipeline is unchanged.

    Args:
        filtered_hotwords: List of filtered hotwords (surface forms).
        transcription: Transcription text (if provided, EOS ``</s>`` is appended).
        language: ``"en"`` (default) or ``"ja"``. Selects the instruction text,
            hotword separator, and reading-annotation behaviour.
        readings: Optional list of readings parallel to ``filtered_hotwords``.
            When given (Japanese), each hotword is annotated as ``橋（ハシ）`` so
            the model can tell same-reading/different-kanji homophones apart.

    Returns:
        prompt: Prompt text in Vicuna chat template format.
    """
    # Build answer part with EOS token
    answer = ""
    if transcription:
        answer = f"{transcription}</s>"

    if language == "ja":
        if not filtered_hotwords:
            prompt = f"USER: 音声を文字に書き起こしてください。\n ASSISTANT:{answer}"
        else:
            hotwords_str = _format_hotwords(filtered_hotwords, readings, "、")
            prompt = (
                f"USER: 音声を文字に書き起こしてください。"
                f"次のキーワードが役立つかもしれません：「{hotwords_str}」。"
                f"\n ASSISTANT:{answer}"
            )
        return prompt

    # Default: English (Vicuna chat format, matching reference)
    if not filtered_hotwords:
        prompt = f"USER: Transcribe speech to text.\n ASSISTANT:{answer}"
    else:
        hotwords_str = _format_hotwords(filtered_hotwords, readings, " ")
        prompt = (
            f"USER: Transcribe speech to text. Some hotwords might help. "
            f'The hotwords are "{hotwords_str}".\n ASSISTANT:{answer}'
        )

    return prompt
