"""Pick the dataset class for a ``(data_dir, split)`` pair.

Two on-disk layouts are supported:

* ``librispeech`` – ``<data_dir>/<split>/<spk>/<chapter>/*.trans.txt`` + ``.flac``
* ``csj``         – ESPnet ``dump/raw`` (Kaldi) layout:
  ``<data_dir>/<split>/{text,wav.scp,utt2spk,utt2num_samples}``

``auto`` picks ``csj`` when ``<data_dir>/<split>/wav.scp`` exists, else
``librispeech``. Importing this module does not import torch, so light scripts
(e.g. ``create_bias_list.py``) can use :func:`resolve_dataset_type` and
:func:`read_kaldi_text`.
"""

from pathlib import Path
from typing import Any, Dict, Optional

DATASET_TYPES = ("auto", "librispeech", "csj")


def resolve_dataset_type(data_dir: str, split: str, dataset_type: str = "auto") -> str:
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"Unknown dataset type {dataset_type!r}; choose from {DATASET_TYPES}")
    if dataset_type != "auto":
        return dataset_type
    return "csj" if (Path(data_dir) / split / "wav.scp").exists() else "librispeech"


def build_dataset(
    data_dir: str,
    split: str,
    dataset_type: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    max_samples: Optional[int] = None,
):
    """Instantiate the dataset for *split* and fail loudly if it is empty.

    ``max_samples`` only applies to CSJ (drop utterances longer than this many
    samples); ``None`` keeps :class:`CSJCorpusDataset`'s 30 s default, ``0``
    keeps every utterance (use this for evaluation).
    """
    kind = resolve_dataset_type(data_dir, split, dataset_type)
    if kind == "csj":
        from src.dataset.kaldi_dataset import CSJCorpusDataset

        kwargs = {} if max_samples is None else {"max_samples": max_samples}
        dataset = CSJCorpusDataset(split=split, base_dir=str(data_dir), **kwargs)
    else:
        from src.dataset.librispeech_dataset import LibriSpeechDataset

        dataset = LibriSpeechDataset(data_path=str(data_dir), split=split, config=config or {})

    if len(dataset) == 0:
        raise RuntimeError(
            f"No utterances loaded from {Path(data_dir) / split} (dataset={kind}). "
            "Check --data-dir / --splits / --dataset."
        )
    return dataset


def read_kaldi_text(path: str) -> Dict[str, str]:
    """Read a Kaldi ``text`` file (``utt_id transcript``) into a dict."""
    transcripts: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            utt, *txt = line.strip().split(maxsplit=1)
            if txt and txt[0].strip():
                transcripts[utt] = txt[0]
    return transcripts
