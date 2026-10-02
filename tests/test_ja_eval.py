"""Tests for Japanese (CSJ) evaluation plumbing: CER scoring, dataset
selection, Kaldi transcript loading, and Whisper-CTC padded-frame masking."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pyscripts"))

from src.dataset.factory import read_kaldi_text, resolve_dataset_type  # noqa: E402
from src.utils.compute_wer import (  # noqa: E402
    compute_cer,
    normalize_japanese,
    score_transcripts,
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_normalize_japanese_strips_punct_space_and_width():
    assert normalize_japanese("東京、 に行く。") == "東京に行く"
    assert normalize_japanese("Ｆ０パターン") == "f0パターン"  # NFKC + lower
    assert normalize_japanese("ラーメン") == "ラーメン"  # long-vowel mark kept


def test_compute_cer_is_corpus_level():
    # 1 substitution over 4 + 0 over 2 reference chars
    assert compute_cer(["abxd", "ef"], ["abcd", "ef"]) == pytest.approx(1 / 6)
    # empty reference does not raise
    assert compute_cer(["a"], [""]) == 1.0


def test_score_ja_uses_cer_not_wer():
    metric, err, hyps, refs = score_transcripts(
        ["東京に行く。"], ["東京に行った。"], language="ja"
    )
    assert metric == "cer"
    assert hyps == ["東京に行く"] and refs == ["東京に行った"]
    # 1 substitution + 1 deletion over 6 reference chars
    assert err == pytest.approx(2 / 6)


def test_score_en_uses_wer():
    metric, err, _, _ = score_transcripts(["a b c"], ["a b d"], language="en")
    assert metric == "wer" and err == pytest.approx(1 / 3)


def test_score_empty_raises_instead_of_zero():
    with pytest.raises(ValueError):
        score_transcripts([], [], language="ja")


# ---------------------------------------------------------------------------
# Dataset selection / Kaldi transcripts
# ---------------------------------------------------------------------------
def _make_kaldi_split(root: Path, split: str) -> Path:
    d = root / split
    d.mkdir(parents=True)
    (d / "text").write_text("u1 内容としましては\nu2\nu3 橋を渡る\n", encoding="utf-8")
    (d / "wav.scp").write_text("u1 dump/raw/x/u1.flac\n", encoding="utf-8")
    return d


def test_resolve_dataset_type(tmp_path):
    _make_kaldi_split(tmp_path, "eval1")
    (tmp_path / "test-clean").mkdir()
    assert resolve_dataset_type(str(tmp_path), "eval1") == "csj"
    assert resolve_dataset_type(str(tmp_path), "test-clean") == "librispeech"
    assert resolve_dataset_type(str(tmp_path), "eval1", "librispeech") == "librispeech"
    with pytest.raises(ValueError):
        resolve_dataset_type(str(tmp_path), "eval1", "bogus")


def test_read_kaldi_text_skips_empty(tmp_path):
    d = _make_kaldi_split(tmp_path, "eval1")
    assert read_kaldi_text(str(d / "text")) == {"u1": "内容としましては", "u3": "橋を渡る"}


def test_create_bias_list_loads_csj_transcripts(tmp_path):
    cbl = pytest.importorskip("create_bias_list")
    _make_kaldi_split(tmp_path, "eval1")
    assert set(cbl.load_transcripts(str(tmp_path), "eval1")) == {"u1", "u3"}


def test_build_dataset_empty_raises(tmp_path):
    pytest.importorskip("torchaudio")
    from src.dataset.factory import build_dataset

    d = tmp_path / "empty"
    d.mkdir()
    (d / "wav.scp").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError):
        build_dataset(str(tmp_path), "empty", "csj")


# ---------------------------------------------------------------------------
# Whisper-CTC: padded frames must decode to blank
# ---------------------------------------------------------------------------
def test_blank_padded_frames():
    torch = pytest.importorskip("torch")
    W = pytest.importorskip("src.finetune.whisper_ft_ctc")

    logits = torch.randn(2, 6, 5)
    lengths = torch.tensor([3, 6])
    out = W.blank_padded_frames(logits, lengths, blank_id=4)
    ids = out.argmax(-1)
    assert (ids[0, 3:] == 4).all()  # padded frames -> blank
    assert torch.equal(out[0, :3], logits[0, :3])  # valid frames untouched
    assert torch.equal(out[1], logits[1])  # full-length utterance untouched


# ---------------------------------------------------------------------------
# Whisper-CTC checkpoint as the LLM model's encoder
# ---------------------------------------------------------------------------
def test_whisper_ctc_base_detection(tmp_path):
    pytest.importorskip("transformers")
    from src.models.encoder import WHISPER_CTC_CONFIG, whisper_ctc_base

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    assert whisper_ctc_base(str(hf_dir)) is None  # regular HF model dir
    assert whisper_ctc_base("openai/whisper-medium") is None  # hub id

    ctc_dir = tmp_path / "checkpoint-1"
    ctc_dir.mkdir()
    (ctc_dir / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError):  # weights but no metadata
        whisper_ctc_base(str(ctc_dir))
    (ctc_dir / WHISPER_CTC_CONFIG).write_text('{"whisper_id": "openai/whisper-medium"}')
    assert whisper_ctc_base(str(ctc_dir)) == "openai/whisper-medium"
