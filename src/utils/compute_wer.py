from typing import List
import logging
import jiwer


logger = logging.getLogger(__name__)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Tính Word Error Rate (WER).

    Args:
        predictions: Danh sách các phiên âm dự đoán
        references: Danh sách các phiên âm ground truth

    Returns:
        wer: Word Error Rate
    """
    try:

        wer = jiwer.wer(references, predictions)
        return wer
    except ImportError:
        raise ImportError(
            "jiwer is required for WER computation. Install with: pip install jiwer"
        )
    except Exception as e:
        logger.error(f"Error computing WER: {e}")
        raise
