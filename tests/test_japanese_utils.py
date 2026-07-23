"""Tests for src.utils.japanese (Japanese text utilities).

These run without torch/transformers — only the lightweight Japanese stack
(pykakasi/jaconv optional, char fallback always available).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.japanese import (  # noqa: E402
    JapaneseTokenizer,
    normalize_reading,
    to_katakana,
    is_kana,
    contains_kanji,
    contains_japanese,
    has_reading_backend,
)

# Kanji -> kana conversion needs fugashi / SudachiPy / pykakasi.
needs_reading = pytest.mark.skipif(
    not has_reading_backend(),
    reason="no kanji->kana backend installed (pykakasi/fugashi/sudachipy)",
)


def test_normalize_reading_folds_hiragana_and_katakana():
    # はし / ハシ / ﾊｼ must all collapse to the same katakana key.
    assert normalize_reading("はし") == normalize_reading("ハシ")
    assert normalize_reading("はし") == "ハシ"


def test_is_kana_and_kanji_detection():
    assert is_kana("はし")
    assert is_kana("ハシ")
    assert not is_kana("橋")
    assert contains_kanji("橋本")
    assert not contains_kanji("はしもと")
    assert contains_japanese("橋")
    assert not contains_japanese("bridge")


def test_tokenizer_segments_text():
    tok = JapaneseTokenizer()
    tokens = tok.segment("東京都に住んでいる")
    # Segmentation must be lossless regardless of which backend is active.
    assert "".join(t.surface for t in tokens) == "東京都に住んでいる"
    assert len(tokens) >= 2


@needs_reading
def test_homophones_share_reading():
    # 同音異字: different kanji, identical reading.
    assert to_katakana("橋") == to_katakana("箸")
    assert to_katakana("橋") == "ハシ"


def test_char_backend_always_available():
    tok = JapaneseTokenizer(backend="char")
    tokens = tok.segment("あい")
    assert [t.surface for t in tokens] == ["あ", "い"]


def test_kana_reading_works_without_any_backend():
    # Even with no analyzer, kana normalisation must work (this is what keeps
    # the char fallback useful for kana-CTC output).
    assert to_katakana("はしもと") == "ハシモト"


def test_empty_input():
    tok = JapaneseTokenizer()
    assert tok.segment("") == []
    assert to_katakana("") == ""
    assert normalize_reading("") == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
