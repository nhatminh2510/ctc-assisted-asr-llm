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

Function words (助詞, 助動詞, 記号 …) and `COMMON_WORDS_PATH` entries are removed
from the hypothesis **only when it contains kanji**. A kana-only CTC run is
mis-segmented by the analyzer (ツカム → ツカ/ム, パターン → タパ/ターン), so removal
would cut syllables out of real words. On CSJ eval1–3 (kana-CTC hypotheses)
keeping the run whole raises hotword recall from 75/72/79% to 91/91/92% while
precision drops by only ~1 point.

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

The chat scaffold follows the LLM's tokenizer (`src/utils/create_prompt.py`):
the Vicuna `USER:/ASSISTANT:` + `</s>` format above for Vicuna/LLaMA-2 models,
and the Llama-3 header format for Llama-3-family decoders (e.g.
`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`), whose answers end with
`<|eot_id|>`:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

音声を文字に書き起こしてください。次のキーワードが役立つかもしれません：「橋（ハシ）、…」。<|eot_id|><|start_header_id|>assistant<|end_header_id|>

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

## The CTC model that feeds the filter

The hotword filter runs on a CTC hypothesis. Two options:

* **Off-the-shelf**: any Japanese CTC model (e.g.
  `imprt/kushinada-hubert-large-laborotv2-asr`) via `decode_ctc.py`
  (`AutoModelForCTC`).
* **Whisper-encoder kana-CTC** (`src/finetune/whisper_ft_ctc.py`): take
  Whisper's encoder, add a CTC head, and fine-tune it against **kana** targets
  so the model emits a pure-kana hypothesis directly — small, fast, and a
  natural pseudo-label source for the reading-based filter. There is no
  `WhisperForCTC` in transformers, so the script defines `WhisperEncoderForCTC`
  (encoder → dropout → Linear → `F.ctc_loss`, blank = pad id) and trains it with
  a HF `Trainer`. Kana targets come from `to_katakana`, so run it in a container
  with a kana backend:

  ```bash
  singularity exec --writable-tmpfs --nv --bind $BINDS --no-home \
      /home/25/minh/Prefix-tree/prefix-tree.sif \
      bash scripts/run-ft-whisper-ctc.sh          # configs/tuning/whisper-ctc-ja.yaml
  ```

  `WhisperEncoderForCTC` is a plain `nn.Module` (no `config.json`), so
  `AutoModelForCTC` cannot load it. `decode_ctc.py` detects such a directory
  (`final/` or `checkpoint-*`) and loads it with `load_whisper_ctc`. New runs
  write `whisper_ctc_config.json` + the vocab into every checkpoint; for
  checkpoints saved before that, pass `BASE_MODEL=openai/whisper-medium`.

  Whisper always encodes a 30 s window. Frames past the end of the audio never
  receive CTC loss, so their argmax is arbitrary; the model therefore forces
  those frames to blank whenever an `attention_mask` is given. Without this,
  greedy decoding over all 1500 frames appends garbage and the reported CER
  exceeds 100% (the `eval_cer` logged by the 70k-step run was affected: ~1.1–1.4,
  while the same `checkpoint-40000` decodes CSJ eval1/2/3 at 5.9 / 7.9 / 7.2%
  kana CER).

## Metrics

Japanese has no spaces, so whitespace WER is meaningless. With `language: ja` (training
config) or `LANGUAGE=ja` (inference / CTC decode) the scripts report **CER**
after NFKC + removing whitespace/punctuation (`src/utils/compute_wer.py`).
`decode_ctc.py` additionally reports `cer_kana` (CER on katakana readings),
the right number for a kana-only CTC model; its `cer_surface` against the
kanji+kana reference is >100% by design. Summaries keep the key `"wer"` for
compatibility and add `"metric": "cer"`. Training logs `eval/cer` to
TensorBoard and saves the best model as `checkpoints/best_cer_model.pt`, which
`run-inference.sh` / `run-all-infer.sh` pick up by default when `LANGUAGE=ja`.

An empty dataset (wrong `--data-dir` / `--splits`) is now an error instead of
a silent 0% error rate.

### WER / U-WER / B-WER (R-WER)

Whenever a bias list is passed (`--bias-list-json`, in any mode including
`nobias`), inference also reports the contextual-biasing metrics of
[is21_deep_bias](https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias)
(`src/utils/bias_wer.py`, a port of its `score.py`: one alignment per utterance
with sub = 4, ins = del = 3, identical results on the reference's own files):

* **B-WER** (= R-WER, rare-word WER): errors on the reference words that are
  biasing words of that utterance (`true_hotwords`), plus insertions of biasing
  words;
* **U-WER**: errors on all other words;
* **WER**: all words.

They are logged per split and saved under `"bias_wer"` in
`<split>_summary.json` and `summary.json`.

For Japanese, "words" come from the morphological analyzer
(`inference.word_backend` in `configs/inference_ja.yaml` or `--word-backend`:
`auto` = fugashi > sudachipy > pykakasi > char) applied to the CER-normalised
text. Consecutive morphemes that spell a biasing word are merged back into one
word, so each biasing word counts once, like an English rare word. Use the same
backend that built the bias list (`create_bias_list.py` uses `auto`, i.e.
fugashi + unidic-lite in both containers): with fugashi, 1 of the 3091 CSJ eval
biasing words fails to align with a word boundary, while SudachiPy's long (mode
C) units hide ~360 of them inside longer words. The count is reported as
`unmatched_biasing_words`.

Re-score saved results without re-running inference:

```bash
H=data/lexicon/csj/hotwords
python3 pyscripts/score_bias_wer.py --language ja \
    --details exp/whisper-ft-ctc_llm_contextual/inference/filter-ja \
    --bias-list-json $H/eval{1,2,3}_whisper_like_clean_bias_list.json \
    --alignment-out /tmp/align.jsonl   # optional: per-utterance words + alignment
```

Score the hotword filter alone — no LLM, no GPU, seconds per split — on the
cached CTC decodes: recall / precision of the picked hotwords against
`true_hotwords` (exact match, summed over utterances). Precision counts the
picks of utterances without any true hotword (72% of eval1), so compare it
between settings rather than reading it as an absolute number. Flags override
the `hotword_filter` section of `--infer-config`:

```bash
python3 pyscripts/eval_hotword_filter.py --filter-method ja \
    --infer-config configs/inference_ja.yaml \
    --common-words-path data/lexicon/csj/common_words_5k.txt \
    --ctc-file exp/whisper-ft-ctc_ja/ctc_decode/eval{1,2,3}_whisper_like_clean_ctc.tsv \
    --bias-list-json $H/eval{1,2,3}_whisper_like_clean_bias_list.json \
    --misses-out /tmp/misses.jsonl     # optional: utterances that missed a true hotword
```

Readings depend on the container: with pykakasi installed
(`ctc-asr-llm_ja_cuda12.4.sif`) recall is 91.6 / 91.1 / 91.6% on eval1/2/3,
without it (`prefix-tree.sif`) 90.8 / 90.7 / 91.5%.

## End-to-end (Japanese, CSJ)

CSJ is read from the ESPnet `dump/raw` layout (`data/csj` symlink, Kaldi
`text` / `wav.scp`); `--dataset auto` picks it when `<split>/wav.scp` exists.
Test sets: `eval{1,2,3}_whisper_like_clean` (1230 / 1113 / 1308 utterances).

```bash
EVAL="eval1_whisper_like_clean eval2_whisper_like_clean eval3_whisper_like_clean"
SIF=/home/25/minh/Prefix-tree/prefix-tree.sif
BINDS=/home,/home-local,/CDshare,/kailab,/App,/LabWork,/UserTmp,/local,/Mirror
run() { singularity exec --writable-tmpfs --nv --bind $BINDS --no-home $SIF "$@"; }

# 1. CTC pseudo-labels (Whisper kana-CTC; or WAVLM_MODEL=imprt/kushinada-hubert-large-laborotv2-asr)
run env WAVLM_MODEL=exp/whisper-ft-ctc_ja/checkpoint-40000 BASE_MODEL=openai/whisper-medium \
    DATA_DIR=data/csj LANGUAGE=ja OUTPUT_DIR=exp/whisper-ft-ctc_ja/ctc_decode \
    SPLITS="$EVAL" bash scripts/decode-ctc.sh

# 2. Bias lists (word frequencies from the CSJ train split)
run env DATA_DIR=data/csj LANGUAGE=ja LEXICON_DIR=data/lexicon/csj \
    OUTPUT_DIR=data/lexicon/csj/hotwords \
    WORD_FREQ_PATH=data/lexicon/csj/all_words.count.txt \
    COMMON_WORDS_PATH=data/lexicon/csj/common_words_5k.txt \
    ALL_RARE_WORDS_PATH=data/lexicon/csj/all_rare_words.txt \
    FREQ_SPLITS=train_nodup_whisper_like_clean TEST_SPLITS="$EVAL" \
    bash scripts/make-bias-list.sh

# 3. Train the Whisper + LLM projector on CSJ (config sets dataset/splits)
run python3 src/train.py --config configs/train_whisper_llm_ja_contextual.yaml \
    --data-dir data/csj --output-dir exp/whisper_llm_ja

# 4. Filtered inference, scored with CER
H=data/lexicon/csj/hotwords; C=exp/whisper-ft-ctc_ja/ctc_decode
run env EXP_DIR=exp/whisper_llm_ja DATA_DIR=data/csj SPLITS="$EVAL" \
    INFER_TYPE=filter FILTER_METHOD=ja LANGUAGE=ja ENCODER_TYPE=whisper \
    INFER_CONFIG=configs/inference_ja.yaml \
    COMMON_WORDS_PATH=data/lexicon/csj/common_words_5k.txt \
    BIAS_LIST_JSON="$(for s in $EVAL; do printf '%s ' $H/${s}_bias_list.json; done)" \
    CTC_FILE="$(for s in $EVAL; do printf '%s ' $C/${s}_ctc.tsv; done)" \
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
