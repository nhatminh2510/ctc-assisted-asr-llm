"""Tests for the kana target normalisation used by Whisper-CTC fine-tuning.

``expand_numbers`` is pure and always runs. ``text_to_kana`` needs a reading
backend (fugashi/SudachiPy/pykakasi) and torch/transformers (for the module
import), so it is skipped when either is missing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The module imports torch/transformers; skip the whole file if unavailable.
W = pytest.importorskip("src.finetune.whisper_ft_ctc")
from src.utils.japanese import has_reading_backend  # noqa: E402

needs_reading = pytest.mark.skipif(
    not has_reading_backend(),
    reason="no kanji->kana backend installed (pykakasi/fugashi/sudachipy)",
)


# ---------------------------------------------------------------------------
# expand_numbers: pure, deterministic (Arabic/full-width digits -> kanji)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("45", "四十五"),
        ("４５", "四十五"),          # full-width
        ("100", "百"),
        ("2024年", "二千二十四年"),
        ("15から", "十五から"),
        ("ＤＩＣ1", "ＤＩＣ一"),
        ("09012345678", "〇九〇一二三四五六七八"),  # ID -> digit-by-digit
        ("no digits", "no digits"),
    ],
)
def test_expand_numbers(text, expected):
    assert W.expand_numbers(text) == expected


# ---------------------------------------------------------------------------
# text_to_kana: numbers survive as their kana reading, output is kana-only
# ---------------------------------------------------------------------------
@needs_reading
def test_numbers_survive_as_kana():
    out = W.text_to_kana("４５ｄＢＡ")
    # 45 is preserved, not dropped. 四十五 has two valid readings and the
    # backend decides which: fugashi -> ヨンジュウゴ, pykakasi -> シジュウゴ.
    assert "ヨンジュウゴ" in out or "シジュウゴ" in out
    # Output must be kana-only (no Arabic/full-width digits left).
    assert all(W._is_kana_char(c) for c in out)


@needs_reading
def test_text_to_kana_is_kana_only():
    out = W.text_to_kana("東京都に住んでいる橋本さん")
    assert out and all(W._is_kana_char(c) for c in out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
