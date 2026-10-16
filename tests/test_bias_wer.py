"""Tests for WER / U-WER / B-WER scoring (port of is21_deep_bias/score.py)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.bias_wer import (  # noqa: E402
    BiasWERScorer,
    Code,
    align,
    merge_biasing_words,
)
from src.utils.japanese import JapaneseTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def test_align_codes_and_indices():
    pairs = align("a b c d".split(), "a x c e d".split())
    assert pairs == [
        (Code.match, 0, 0),
        (Code.substitution, 1, 1),
        (Code.match, 2, 2),
        (Code.insertion, None, 3),
        (Code.match, 3, 4),
    ]
    assert align(["a", "b"], ["b"]) == [(Code.deletion, 0, None), (Code.match, 1, 0)]
    assert align([], []) == []


def test_align_prefers_substitution_over_ins_plus_del():
    # sub costs 4 < ins + del = 6
    assert [c for c, _, _ in align(["a"], ["b"])] == [Code.substitution]


# ---------------------------------------------------------------------------
# U-WER / B-WER attribution (English, whitespace words)
# ---------------------------------------------------------------------------
def test_u_wer_b_wer_split():
    scorer = BiasWERScorer("en")
    res = scorer.score(
        ["the quick fox jumped", "hello kafka"],
        ["the quick brown fox jumped", "hello world"],
        [["brown"], ["world"]],
    )
    # utt 1: "brown" deleted (biased); utt 2: world -> kafka (biased sub)
    assert (res.wer.ref_words, res.wer.errors) == (7, 2)
    assert (res.b_wer.ref_words, res.b_wer.dels, res.b_wer.subs) == (2, 1, 1)
    assert (res.u_wer.ref_words, res.u_wer.errors) == (5, 0)
    assert res.b_wer.rate == pytest.approx(1.0)
    assert res.unmatched_biasing_words == 0


def test_insertion_of_biasing_word_counts_in_b_wer():
    res = BiasWERScorer("en").score(["a kafka b"], ["a b"], [["kafka"]])
    assert (res.b_wer.ins, res.b_wer.ref_words, res.b_wer.rate) == (1, 0, None)
    assert res.u_wer.errors == 0 and res.wer.ins == 1


def test_results_add_up_across_splits():
    scorer = BiasWERScorer("en")
    a = scorer.score(["a x"], ["a b"], [["b"]])
    b = scorer.score(["c"], ["c d"], [[]])
    total = a + b
    assert total.wer.ref_words == 4 and total.wer.errors == 2
    assert total.b_wer.subs == 1 and total.u_wer.dels == 1 and total.num_utts == 2


def test_biasing_words_are_normalised_like_text():
    lower = lambda s: s.lower()  # e.g. LibriSpeech bias lists are upper case
    res = BiasWERScorer("en", normalizer=lower).score(["Hello Wrld"], ["HELLO WORLD"], [["WORLD"]])
    assert res.b_wer.ref_words == 1 and res.b_wer.subs == 1


# ---------------------------------------------------------------------------
# Merging biasing words split into several tokens
# ---------------------------------------------------------------------------
def test_merge_biasing_words_on_token_boundaries():
    toks = ["生成", "過程", "モデル", "の", "藤崎"]
    assert merge_biasing_words(toks, {"生成過程モデル", "藤崎"}) == ["生成過程モデル", "の", "藤崎"]
    # never carved out of a longer token
    assert merge_biasing_words(["東京都", "に"], {"京都"}) == ["東京都", "に"]
    # longest match wins
    assert merge_biasing_words(["a", "b", "c"], {"a b", "a b c"}, " ") == ["a b c"]


# ---------------------------------------------------------------------------
# Japanese: morphological-analyzer words
# ---------------------------------------------------------------------------
def _has(backend):
    return JapaneseTokenizer(backend).backend_name == backend


@pytest.mark.skipif(not _has("fugashi"), reason="fugashi not installed")
def test_japanese_words_with_fugashi():
    scorer = BiasWERScorer("ja", ja_backend="fugashi")
    assert scorer.split.name == "fugashi"
    ref = "生成過程モデル藤崎モデルなんですが"
    hyp, ref_t, bias = scorer.tokenize(ref, ref, ["生成過程モデル", "藤崎"])
    assert "生成過程モデル" in ref_t and "藤崎" in ref_t  # one word each
    assert len(ref_t) > 3  # but the rest is really segmented, not one chunk

    res = scorer.score(
        ["生成過程モデル藤咲モデルなんですが。", ref],
        [ref, ref],
        [["生成過程モデル", "藤崎"]] * 2,
    )
    assert res.b_wer.ref_words == 4 and res.b_wer.subs == 1
    assert res.u_wer.subs == 0  # punctuation is normalised away
    assert res.unmatched_biasing_words == 0


@pytest.mark.skipif(not _has("sudachipy"), reason="sudachipy not installed")
def test_japanese_words_with_sudachipy():
    scorer = BiasWERScorer("ja", ja_backend="sudachipy")
    res = scorer.score(["藤崎モデルです"], ["藤崎モデルです"], [["藤崎"]])
    assert res.wer.errors == 0 and res.wer.ref_words > 1
