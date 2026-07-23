"""Tests for the Japanese hotword filter.

Each test targets one of the three Japanese-specific problems the filter is
designed to solve. They run without torch/transformers.

Tests that require converting kanji hotwords into readings are skipped when no
analyzer (fugashi / SudachiPy / pykakasi) is installed, since without one a
kanji hotword genuinely cannot be matched against a kana hypothesis.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.hotword_filter_ja import HotwordFilter  # noqa: E402
from src.utils.japanese import has_reading_backend  # noqa: E402

needs_reading = pytest.mark.skipif(
    not has_reading_backend(),
    reason="no kanji->kana backend installed (pykakasi/fugashi/sudachipy)",
)


def _filter(**kwargs):
    return HotwordFilter(word_num=15, **kwargs)


# ---------------------------------------------------------------------------
# Problem 1: no word boundaries (a kana CTC run has no spaces to split on)
# ---------------------------------------------------------------------------
@needs_reading
def test_no_whitespace_kana_run():
    f = _filter()
    # A continuous kana hypothesis (typical kana-CTC output).
    hyp = "とうきょうとにすんでいるはしもとさんにあいました"
    bias = ["橋本", "山田", "田中", "大阪府", "名古屋"]
    assert "橋本" in f.filter_hotwords(hyp, bias)


def test_no_whitespace_kana_bias_list():
    # Same problem, but with a kana bias list: works with *any* backend
    # because no kanji->kana conversion is needed.
    f = _filter()
    hyp = "とうきょうとにすんでいるはしもとさんにあいました"
    bias = ["ハシモト", "ヤマダ", "オオサカ"]
    assert "ハシモト" in f.filter_hotwords(hyp, bias)


# ---------------------------------------------------------------------------
# Problem 2: homophones — same reading, different kanji (同音異字)
# ---------------------------------------------------------------------------
@needs_reading
def test_homophones_retrieved_from_kana_hypothesis():
    f = _filter(include_homophones=True, max_homophones=3)
    hyp = "かれははしをつかった"  # kana-only: cannot tell 箸 from 橋
    bias = ["箸", "橋", "猫", "本", "机"]
    pairs = f.filter_hotwords_with_readings(hyp, bias)
    surfaces = [s for s, _ in pairs]
    # Both same-reading kanji should be surfaced for the LLM to disambiguate.
    assert "箸" in surfaces
    assert "橋" in surfaces
    # And every returned hotword carries its (shared) reading.
    assert all(r == "ハシ" for _, r in pairs)


@needs_reading
def test_kanji_hypothesis_prefers_matching_surface():
    # When the hypothesis itself has kanji, the exact surface should rank first.
    f = _filter(include_homophones=False)
    out = f.filter_hotwords("彼は箸を使った", ["箸", "橋", "端"])
    assert out[0] == "箸"


# ---------------------------------------------------------------------------
# Problem 3: polyphony — a kanji read differently in context (多音読み)
# ---------------------------------------------------------------------------
@needs_reading
def test_polyphony_context_reading():
    f = _filter()
    # 生 is read なま here; matching is on the reading, so 生ビール is found.
    hyp = "なまビールをのむ"
    bias = ["生ビール", "青春", "一生", "人生"]
    assert "生ビール" in f.filter_hotwords(hyp, bias)


# ---------------------------------------------------------------------------
# General behaviour (backend-independent)
# ---------------------------------------------------------------------------
@needs_reading
def test_multiple_true_hotwords():
    f = _filter()
    hyp = "やまだせんせいとさとうくんがとうきょうにいった"
    bias = ["山田", "佐藤", "東京", "大阪", "名古屋"]
    out = set(f.filter_hotwords(hyp, bias))
    assert {"山田", "佐藤", "東京"}.issubset(out)


def test_word_num_cap():
    f = HotwordFilter(word_num=2, include_homophones=False)
    hyp = "やまだとさとうとたなかとすずき"
    bias = ["ヤマダ", "サトウ", "タナカ", "スズキ"]
    assert len(f.filter_hotwords(hyp, bias)) <= 2


def test_empty_inputs():
    f = _filter()
    assert f.filter_hotwords("", ["橋"]) == []
    assert f.filter_hotwords("はし", []) == []


def test_no_match_returns_empty():
    f = _filter()
    # Unrelated bias list -> nothing above threshold.
    assert f.filter_hotwords("こんにちは", ["リョウシリキガク", "ビブンホウテイシキ"]) == []


def test_interface_compat_returns_surfaces():
    f = _filter()
    out = f.filter_hotwords("はしもとさん", ["ハシモト"])
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)


def test_threshold_is_respected():
    # A very high threshold should reject a merely-similar candidate.
    f = HotwordFilter(word_num=15, probability_threshold=0.99, include_homophones=False)
    assert f.filter_hotwords("やまぐち", ["ヤマダ"]) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
