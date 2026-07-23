# Japanese contextual ASR & the Whisper encoder

This adds two things to the CTC-assisted LLM ASR pipeline:

1. **A Whisper option for the speech encoder** (in addition to WavLM / HuBERT /
   wav2vec2), selectable per experiment.
2. **A Japanese hotword filter** (`ja`) that selects contextual hotwords from a
   bias list and injects them — with their readings — into the LLM prompt,
   handling the three problems that break the English filters.

Nothing about the English pipeline changes; both are opt-in.

---

## 1. Whisper as the encoder

The encoder is now wrapped by `src/models/encoder.py::SpeechEncoder`, which
exposes one small interface (`process_audio` → `extract_features` →
`output_lengths`) for two very different encoder families:

| `encoder_type` | Model class            | Input            | Notes                                  |
| -------------- | ---------------------- | ---------------- | -------------------------------------- |
| `wavlm` / `hubert` / `wav2vec2` | `AutoModelForCTC` | raw waveform (`input_values`) | uses `_get_feat_extract_output_lengths` |
| `whisper`      | `WhisperModel.encoder` | log-mel (`input_features`) | mel frames downsampled ×2, 30 s window  |

`encoder_type` defaults to `auto`, which detects the family from the checkpoint
config (or the name). Set it explicitly in a config or on the CLI:

```yaml
model:
  encoder_path: openai/whisper-large-v3
  encoder_type: whisper          # or: auto
  llm_model_path: tokyotech-llm/Swallow-7b-instruct-hf
```

```bash
# inference
ENCODER_TYPE=whisper LANGUAGE=ja bash scripts/run-inference.sh
```

The projector, LLM, masking and training loop are unchanged — only the encoder
tower and its feature front-end differ. Whisper always pads audio to 30 s and
emits 1500 frames; the wrapper computes the number of *valid* frames from the
feature attention mask so padded frames are masked out of the LLM context.

See `configs/train_whisper_llm_ja_contextual.yaml` for a full training config.

---

## 2. Japanese hotword filtering (`--filter-method ja`)

Japanese breaks the character/whitespace filters (`F1`–`F3`) in three ways.
`src/inference/hotword_filter_ja.py` addresses each one:

### a) No word boundaries (分かち書きが無い)

Japanese has no spaces, so `str.split()` is useless — and a kana-based CTC model
emits one long unsegmented run. The filter:

* segments text with a morphological analyzer
  (`fugashi`/MeCab → `SudachiPy` → `pykakasi` → per-character fallback), and
* matches each hotword's reading as a **fuzzy substring** of the whole
  hypothesis (sliding window), so recall does not depend on perfect
  segmentation.

### b) Polyphony / multiple readings (多音読み)

The same kanji is read differently in context (生 → セイ/ショウ/なま/…). Retrieval
and scoring run on the **reading** (context-resolved katakana), not the surface
kanji, so a hotword matches the pronunciation the CTC model actually produced.

### c) Homophones — same reading, different kanji (同音異字)

橋 / 箸 / 端 are all はし. A reading n-gram index gives high recall for these even
from a kana-only hypothesis. Because a kana hypothesis *cannot* pick the right
kanji on its own, the filter:

* retrieves all same-reading candidates,
* ranks them with `reading_similarity + surface_bonus · surface_similarity`
  (surface only helps when the hypothesis itself contains kanji), and
* emits the homophone set together (`include_homophones`) so the **LLM** makes
  the final choice from the audio.

Each selected hotword is passed to the prompt annotated with its reading:

```
USER: 音声を文字に書き起こしてください。次のキーワードが役立つかもしれません：「橋（ハシ）、箸（ハシ）、橋本（ハシモト）」。
 ASSISTANT:
```

### Key parameters (`configs/inference_ja.yaml`)

| Param                  | Default | Meaning                                             |
| ---------------------- | ------- | --------------------------------------------------- |
| `probability_threshold`| `0.5`   | precision floor on the combined score               |
| `word_num`             | `15`    | max hotwords kept                                   |
| `reading_weight`       | `1.0`   | weight of the reading term                          |
| `surface_bonus`        | `0.3`   | tie-breaker when the hypothesis has kanji           |
| `include_homophones`   | `true`  | emit same-reading variants together                 |
| `max_homophones`       | `3`     | cap on homophones added per selected hotword        |
| `tokenizer_backend`    | `auto`  | `fugashi` / `sudachipy` / `pykakasi` / `char`       |

---

## End-to-end (Japanese)

```bash
# 1. CTC pseudo-labels from a Japanese CTC model (AutoModelForCTC)
WAVLM_MODEL=imprt/kushinada-hubert-large-laborotv2-asr bash scripts/decode-ctc.sh

# 2. Bias lists with Japanese segmentation
python pyscripts/create_bias_list.py --language ja \
    --data-dir <corpus> --output-dir data/hotwords_ja \
    --word-freq-path data/ja_words.count.txt

# 3. Filtered inference: Japanese filter + Japanese prompt + Whisper encoder
INFER_TYPE=filter FILTER_METHOD=ja LANGUAGE=ja ENCODER_TYPE=whisper \
    INFER_CONFIG=configs/inference_ja.yaml \
    BIAS_LIST_JSON="data/hotwords_ja/test_bias_list.json" \
    CTC_FILE="exp/.../jp_ctc_decode/test_ctc.tsv" \
    bash scripts/run-inference.sh
```

## Dependencies & graceful degradation

The filter never hard-fails on a missing dependency:

* best:   `fugashi`+`unidic-lite` or `sudachipy`+`sudachidict-core` — real
  segmentation and context readings;
* good:   `pykakasi` (+`jaconv`) — kanji→reading, lighter segmentation;
* always: a pure-python character fallback — kana↔kana matching still works,
  but **kanji hotwords cannot be matched against a kana hypothesis**, because
  nothing can convert 橋 → ハシ.

### Which container to use

| Container | fugashi / SudachiPy | Japanese filter |
| --------- | ------------------- | --------------- |
| `/home/25/minh/Prefix-tree/prefix-tree.sif` | ✅ both, with dictionaries | fully functional |
| `containers/ctc-asr-llm_cuda12.4.sif`       | ❌ none                    | degraded (char fallback) |

Run the Japanese pipeline in `prefix-tree.sif`, e.g.

```bash
SIF=/home/25/minh/Prefix-tree/prefix-tree.sif
BINDS=/home,/home-local,/CDshare,/kailab,/App,/LabWork,/UserTmp,/local,/Mirror
singularity exec --writable-tmpfs --nv --bind $BINDS --no-home $SIF python3 -m pytest tests/ -q
```

To make `ctc-asr-llm_cuda12.4.sif` usable for Japanese, add `fugashi`,
`unidic-lite` (or `sudachipy` + `sudachidict-core`) and optionally `jaconv` to
its `.def` file and rebuild. In the degraded mode kana↔kana matching still
works, but kanji hotwords cannot be matched against a kana hypothesis, and
`create_bias_list.py --language ja` exits with an explicit error rather than
silently producing empty lists.

Check which backends are live:

```bash
python -c "from src.utils.japanese import backend_info, has_reading_backend; \
           print(backend_info(), has_reading_backend())"
```

## Tests

```bash
pytest tests/ -v
```

Tests that need kanji→kana conversion are **skipped** (not failed) when no
analyzer is installed, so the suite is honest about the degraded mode.
