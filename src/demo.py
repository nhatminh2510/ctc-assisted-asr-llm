"""Worked example / trace for src.inference.hotword_filter_ja."""
import sys
from pathlib import Path

ROOT = Path("/home/25/minh/ctc-assisted-asr-llm")
sys.path.insert(0, str(ROOT))

from src.utils.japanese import backend_info, to_katakana
from src.inference.hotword_filter_ja import HotwordFilter

print("=" * 78)
print("BACKENDS:", backend_info())
print("=" * 78)


def trace(title, hyp, bias, **kw):
    f = HotwordFilter(**kw)
    print("\n" + "#" * 78)
    print("#", title)
    print("#" * 78)
    print(f"tokenizer backend : {f.tokenizer.backend_name}")
    print(f"CTC hypothesis    : {hyp}")
    print(f"bias list ({len(bias)})    : {bias}")

    # --- step 1: index -------------------------------------------------
    entries, r_idx, s_idx, r_exact = f._build_entries(bias)
    print("\n[1] bias-list entries (surface -> reading, reading n-grams)")
    for e in entries:
        print(
            f"    {e.surface:<8} -> {e.reading:<10} ngrams={sorted(e.reading_ngrams)}"
        )
    dupes = {
        r: sorted(entries[i].surface for i in ids)
        for r, ids in r_exact.items()
        if len(ids) > 1
    }
    if dupes:
        print(f"    same-reading groups (同音異字): {dupes}")

    # --- step 2: segment hypothesis ------------------------------------
    all_tokens = f.tokenizer.segment(hyp)
    content = f.extract_content_tokens(hyp)
    print("\n[2] segmentation of the hypothesis")
    print("    all tokens    :", [(t.surface, t.reading, t.pos) for t in all_tokens])
    print("    content tokens:", [(t.surface, t.reading) for t in content])
    hyp_reading, hyp_surface = f._build_hypothesis_strings(hyp)
    print(f"    hyp_reading   : {hyp_reading}")
    print(f"    hyp_surface   : {hyp_surface}")

    # --- step 3: retrieval ---------------------------------------------
    cand = f._retrieve_from_strings(hyp_reading, hyp_surface, r_idx, s_idx)
    print("\n[3] candidates retrieved by n-gram overlap")
    print(
        "    ",
        sorted(entries[c].surface for c in cand),
        f"  ({len(cand)}/{len(entries)} of the bias list)",
    )

    # --- step 4: scoring -------------------------------------------------
    print(
        "\n[4] scores = reading_sim*%.1f + surface_sim*%.1f   (threshold %.2f)"
        % (f.reading_weight, f.surface_bonus, f.probability_threshold)
    )
    rows = []
    for c in cand:
        e = entries[c]
        rs = f._best_window_ratio(e.reading, hyp_reading)
        ss = f._best_window_ratio(e.surface, hyp_surface)
        rows.append(
            (f._score(e, hyp_reading, hyp_surface), e.surface, e.reading, rs, ss)
        )
    for score, surf, read, rs, ss in sorted(rows, reverse=True):
        mark = "KEEP" if score >= f.probability_threshold else "drop"
        print(
            f"    {mark}  {surf:<8} ({read:<8}) score={score:.3f}"
            f"  reading_sim={rs:.3f} surface_sim={ss:.3f}"
        )

    # --- step 5: final ----------------------------------------------------
    out = f.filter_hotwords_with_readings(hyp, bias)
    print("\n[5] result -> prompt annotation")
    print("    ", [f"{s}（{r}）" for s, r in out])
    return out


# ---------------------------------------------------------------------------
# Example 1 — homophones (同音異字): kana-only CTC output, 3 candidates kept
# ---------------------------------------------------------------------------
# trace(
#     "Example 1: homophones — CTC emits pure kana はし",
#     "きょう はし を わたって かいしゃ に いきました",
#     ["橋", "箸", "端", "駅", "会社", "空港", "山田", "田中", "新幹線"],
# )

# ---------------------------------------------------------------------------
# Example 2 — polyphony (多音読み): 生 read differently in hyp vs hotword
# ---------------------------------------------------------------------------
# trace(
#     "Example 2: polyphony — なまビール / 生ビール",
#     "なまびーる を ください",
#     ["生ビール", "生放送", "学生", "先生", "ビール"],
# )

# ---------------------------------------------------------------------------
# Example 3 — no word boundaries: one long unsegmented kana run
# ---------------------------------------------------------------------------
trace(
    "Example 3: no boundaries — unsegmented kana run + ASR error",
    "しんかんせんでしんおおさかまでいきます",
    ["新幹線", "新大阪", "品川", "在来線", "東京駅", "特急"],
    #     "さいきんかまたにいてんになりまして",
    #     ["静岡", "大学", "蒲田", "荻窪", "東京駅", "特急"],
)

# ---------------------------------------------------------------------------
# Example 4 — precision mode: per_word_scoring=True, first=1
# ---------------------------------------------------------------------------
# trace(
#     "Example 4: precision mode (per_word_scoring=True, first=1, no homophones)",
#     "きょう はし を わたって かいしゃ に いきました",
#     ["橋", "箸", "端", "駅", "会社", "空港"],
#     per_word_scoring=True,
#     first=1,
#     include_homophones=False,
# )

# ---------------------------------------------------------------------------
# Example 5 — kanji in the hypothesis: surface_bonus breaks the homophone tie
# ---------------------------------------------------------------------------
# trace(
#     "Example 5: kanji hypothesis — surface bonus picks the right kanji",
#     "今日は箸を買いました",
#     ["橋", "箸", "端", "皿", "茶碗"],
#     per_word_scoring=True,
#     first=1,
#     include_homophones=False,
# )
