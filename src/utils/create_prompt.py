from typing import List
import logging

logger = logging.getLogger(__name__)


def create_prompt_with_hotwords(
    filtered_hotwords: List[str],
    transcription: str = "",
) -> str:
    """
    Create prompt text with filtered hotwords for LLM.
    Matches reference Vicuna chat format: "USER: {prompt}\\n ASSISTANT:{answer}</s>"

    Args:
        filtered_hotwords: List of filtered hotwords
        transcription: Transcription text (if provided, EOS '</s>' is appended)

    Returns:
        prompt: Prompt text in Vicuna chat template format

    Examples:
        filtered_hotwords = ['bertie', 'alice']
        -> 'USER: Transcribe speech to text. Some hotwords might help. The hotwords are "bertie alice".\n ASSISTANT:'

        filtered_hotwords = ['bertie', 'alice'], transcription = "hello world"
        -> 'USER: ... The hotwords are "bertie alice".\n ASSISTANT:hello world</s>'
    """
    # Build answer part with EOS token
    answer = ""
    if transcription:
        answer = f"{transcription}</s>"

    # Build prompt with correct Vicuna chat format (matching reference)
    if not filtered_hotwords:
        prompt = f"USER: Transcribe speech to text.\n ASSISTANT:{answer}"
    else:
        hotwords_str = " ".join(filtered_hotwords)
        prompt = (
            f"USER: Transcribe speech to text. Some hotwords might help. "
            f'The hotwords are "{hotwords_str}".\n ASSISTANT:{answer}'
        )

    return prompt
