"""Japanese text-processing utilities for contextual hotword filtering.

Japanese raises three problems that the English-oriented hotword filters
(``hotword_filter_F1/F2/F3``) do not handle. This module provides the building
blocks used by ``src.inference.hotword_filter_ja`` to solve them:

1. **No word boundaries** (分かち書きが無い).
   Japanese is written without spaces, so ``str.split()`` cannot recover
   words. We segment text with a morphological analyzer (``fugashi``/MeCab or
   ``SudachiPy``). When neither is installed we fall back to ``pykakasi``
   (which performs a lighter segmentation) and finally to a character-level
   split. Retrieval still works in the fallback because it is driven by
   *reading* n-grams, not whole words.

2. **Multiple readings / polyphony** (多音字・多音読み).
   The same kanji can be read differently depending on context, e.g.
   生 → セイ / ショウ / なま / い(きる) …  A morphological analyzer resolves the
   reading from context; we expose that reading (normalised to katakana) so
   that matching is performed on *pronunciation* rather than on surface
   characters.

3. **Homophones / same reading, different characters** (同音異字).
   e.g. 橋 / 箸 / 端 are all read はし.  Reading-based matching gives high recall
   for these; the caller combines a surface-similarity term to keep precision
   and to pick the correct kanji when the CTC hypothesis happens to contain
   kanji.

The module degrades gracefully: every backend is optional, and importing this
file never fails because of a missing dependency.  Call :func:`backend_info`
to see which backends are active at runtime.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kana / script helpers (pure-python, no external dependency)
# ---------------------------------------------------------------------------
_HIRA_START, _HIRA_END = 0x3041, 0x3096
_KATA_START = 0x30A1
_KATA_HIRA_OFFSET = _KATA_START - _HIRA_START  # 0x60

_KANA_RE = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
_JAPANESE_RE = re.compile(r"[぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")
_KANJI_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _hira_to_kata(text: str) -> str:
    """Convert hiragana code points to katakana (pure python)."""
    out = []
    for ch in text:
        code = ord(ch)
        if _HIRA_START <= code <= _HIRA_END:
            out.append(chr(code + _KATA_HIRA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def contains_japanese(text: str) -> bool:
    """Return True if *text* contains any Japanese (kana/kanji) character."""
    return bool(_JAPANESE_RE.search(text or ""))


def contains_kanji(text: str) -> bool:
    """Return True if *text* contains at least one kanji character."""
    return bool(_KANJI_RE.search(text or ""))


def is_kana(text: str) -> bool:
    """Return True if *text* is non-empty and made only of kana characters."""
    text = (text or "").strip()
    if not text:
        return False
    return all(_KANA_RE.match(ch) for ch in text)


# ``jaconv`` gives the most correct full/half-width + kana normalisation, but we
# provide a pure-python fallback so the module works without it.
try:  # pragma: no cover - trivial import guard
    import jaconv  # type: ignore

    _HAS_JACONV = True
except Exception:  # pragma: no cover
    jaconv = None  # type: ignore
    _HAS_JACONV = False


def normalize_reading(text: str) -> str:
    """Normalise a reading to a canonical katakana form.

    Steps: NFKC / width normalisation, hiragana → katakana, and removal of any
    whitespace.  The result is what all reading comparisons and the reading
    n-gram index operate on, so that はし / ハシ / ﾊｼ all collapse to ``ハシ``.
    """
    if not text:
        return ""
    if _HAS_JACONV:
        text = jaconv.normalize(text)  # full/half width, etc.
        text = jaconv.hira2kata(text)
    else:
        text = _hira_to_kata(text)
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------------------
# Reading conversion (kanji -> kana) via pykakasi, with graceful fallback
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    import pykakasi  # type: ignore

    _KKS = pykakasi.kakasi()
    _HAS_PYKAKASI = True
except Exception:  # pragma: no cover
    _KKS = None
    _HAS_PYKAKASI = False


# A word-level analyzer (fugashi/SudachiPy) can also supply readings when
# pykakasi is absent. Resolved lazily and cached; deliberately restricted to the
# analyzer backends so it can never recurse into the character fallback (which
# itself calls :func:`to_katakana`).
_READING_TOKENIZER = None
_READING_TOKENIZER_TRIED = False


def _reading_tokenizer():
    """Return a word-level backend usable for kanji → kana, or ``None``."""
    global _READING_TOKENIZER, _READING_TOKENIZER_TRIED
    if not _READING_TOKENIZER_TRIED:
        _READING_TOKENIZER_TRIED = True
        for name in ("fugashi", "sudachipy"):
            try:
                _READING_TOKENIZER = _BACKEND_CLASSES[name]()
                break
            except Exception:  # pragma: no cover - depends on environment
                continue
    return _READING_TOKENIZER


@lru_cache(maxsize=100_000)
def to_katakana(text: str) -> str:
    """Best-effort conversion of arbitrary Japanese text to a katakana reading.

    Tries ``pykakasi`` first (cheap, dictionary-free), then a morphological
    analyzer (``fugashi``/``SudachiPy``) which also resolves kanji readings from
    context.  With none of them only kana characters are normalised and kanji
    are left as-is (still usable as a fallback key).  Cached because readings
    are looked up repeatedly.
    """
    if not text:
        return ""
    if _HAS_PYKAKASI:
        try:
            kana = "".join(item["kana"] for item in _KKS.convert(text))
            return normalize_reading(kana)
        except Exception:  # pragma: no cover - defensive
            pass
    tokenizer = _reading_tokenizer()
    if tokenizer is not None:
        try:
            reading = "".join(t.reading for t in tokenizer.segment(text))
            if reading:
                return normalize_reading(reading)
        except Exception:  # pragma: no cover - defensive
            pass
    return normalize_reading(text)


def has_reading_backend() -> bool:
    """True if kanji → kana conversion is actually available.

    Without one (no fugashi/SudachiPy/pykakasi) a kanji hotword cannot be
    matched against a kana hypothesis; callers/tests use this to degrade or
    skip rather than silently produce empty results.
    """
    if _HAS_PYKAKASI:
        return True
    for name in ("fugashi", "sudachipy"):
        try:
            __import__(name)
            return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Segmentation backends
# ---------------------------------------------------------------------------
@dataclass
class Token:
    """A segmented Japanese token.

    Attributes:
        surface: The written form as it appears in the text (kanji/kana/…).
        reading: Normalised katakana reading (context-resolved when the backend
            supports it). Falls back to :func:`to_katakana` of the surface.
        pos: Coarse part-of-speech tag (``名詞`` etc.) or ``None`` if unknown.
    """

    surface: str
    reading: str
    pos: Optional[str] = None


class _BaseBackend:
    name = "base"
    available = False

    def segment(self, text: str) -> List[Token]:  # pragma: no cover - abstract
        raise NotImplementedError


class _FugashiBackend(_BaseBackend):
    """MeCab via fugashi. Best quality: context readings + POS."""

    name = "fugashi"

    def __init__(self) -> None:
        import fugashi  # type: ignore

        self._tagger = fugashi.Tagger()
        self.available = True

    def segment(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        for word in self._tagger(text):
            feat = word.feature
            # UniDic exposes ``kana``/``pron``; other dics expose ``reading``.
            reading = (
                getattr(feat, "kana", None)
                or getattr(feat, "pron", None)
                or getattr(feat, "reading", None)
                or word.surface
            )
            pos = getattr(feat, "pos1", None)
            tokens.append(Token(word.surface, normalize_reading(reading), pos))
        return tokens


class _SudachiBackend(_BaseBackend):
    """SudachiPy. Good quality: reading_form() + POS."""

    name = "sudachipy"

    def __init__(self) -> None:
        from sudachipy import dictionary, tokenizer  # type: ignore

        self._tok = dictionary.Dictionary().create()
        self._mode = tokenizer.Tokenizer.SplitMode.C
        self.available = True

    def segment(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        for m in self._tok.tokenize(text, self._mode):
            reading = m.reading_form() or m.surface()
            pos = m.part_of_speech()[0] if m.part_of_speech() else None
            tokens.append(Token(m.surface(), normalize_reading(reading), pos))
        return tokens


class _PykakasiBackend(_BaseBackend):
    """pykakasi. No POS, coarse segmentation, but resolves kanji readings."""

    name = "pykakasi"

    def __init__(self) -> None:
        if not _HAS_PYKAKASI:
            raise RuntimeError("pykakasi is not installed")
        self._kks = _KKS
        self.available = True

    def segment(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        for item in self._kks.convert(text):
            surface = item.get("orig", "")
            if not surface:
                continue
            reading = item.get("kana") or surface
            tokens.append(Token(surface, normalize_reading(reading), None))
        return tokens


class _CharBackend(_BaseBackend):
    """Last-resort: one token per (non-space) character.

    Retrieval is reading-driven, so even a character split lets a kana CTC
    hypothesis reach kanji hotwords through the reading n-gram index.
    """

    name = "char"
    available = True

    def segment(self, text: str) -> List[Token]:
        return [Token(ch, to_katakana(ch), None) for ch in text if not ch.isspace()]


_BACKEND_ORDER = ("fugashi", "sudachipy", "pykakasi", "char")
_BACKEND_CLASSES = {
    "fugashi": _FugashiBackend,
    "sudachipy": _SudachiBackend,
    "pykakasi": _PykakasiBackend,
    "char": _CharBackend,
}


class JapaneseTokenizer:
    """Segment Japanese text into :class:`Token` objects.

    Args:
        backend: ``"auto"`` (default) picks the best available backend in the
            order fugashi → sudachipy → pykakasi → char. Pass an explicit name
            to force one; if it is unavailable we log a warning and fall back
            to auto-selection.
    """

    def __init__(self, backend: str = "auto") -> None:
        self.backend = self._resolve_backend(backend)
        logger.info(
            "JapaneseTokenizer using backend '%s' (requested '%s')",
            self.backend.name,
            backend,
        )

    @staticmethod
    def _try_build(name: str) -> Optional[_BaseBackend]:
        try:
            return _BACKEND_CLASSES[name]()
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.debug("Backend '%s' unavailable: %s", name, exc)
            return None

    def _resolve_backend(self, backend: str) -> _BaseBackend:
        if backend != "auto":
            built = self._try_build(backend)
            if built is not None:
                return built
            logger.warning(
                "Requested Japanese backend '%s' is unavailable; "
                "falling back to auto-selection.",
                backend,
            )
        for name in _BACKEND_ORDER:
            built = self._try_build(name)
            if built is not None:
                return built
        # ``char`` is always available, so we never reach here.
        return _CharBackend()

    def segment(self, text: str) -> List[Token]:
        if not text:
            return []
        return self.backend.segment(text)

    @property
    def backend_name(self) -> str:
        return self.backend.name


def backend_info() -> dict:
    """Return which optional backends are importable (for logging/debugging)."""
    info = {"jaconv": _HAS_JACONV, "pykakasi": _HAS_PYKAKASI}
    for name in ("fugashi", "sudachipy"):
        try:
            __import__(name)
            info[name] = True
        except Exception:
            info[name] = False
    return info
