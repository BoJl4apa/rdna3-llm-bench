# A Hebrew STT lane: ivrit.ai's whisper-large-v3 fine-tune vs stock large-v3 on real dictations

**TL;DR** — On 38 real Hebrew push-to-talk dictations (5–10 s, technical
speech with embedded English product names), **ivrit-ai/whisper-large-v3**
(the Hebrew fine-tune, ggml f16, whisper.cpp HIP on a W7800, beam 5, language
forced to `he`) beat stock **whisper large-v3** on **12 clips, lost on 4, tied
the rest**, at the same ~0.55 s per utterance. It now serves Hebrew as its
own lane; stock large-v3 keeps every other language. The 10-item Hebrew slice
of a scripted dictation corpus could **not** see the difference (44.9 % vs
48.0 % WER, ±4 points of noise on ~100 words) — the decision came from
reading real captures side by side. Turbo, large-v2, prompt tricks and a
hosted frontier API all lost.

## Why a second model at all

Whisper saw ~688 h of Hebrew against 438 k h of English (paper, Fig. 11), its
byte-level tokenizer is a poor fit for Hebrew (the authors name it as an
outlier), and unvocalised Hebrew strips the decoder's language-model prior.
Published Hebrew WER for stock large-v3 is 26 % on FLEURS-he; the ivrit.ai
fine-tune (~5,000 h of Hebrew, Apache-2.0) reports 17 % there and halves the
error on conversational sets. Whether that survives short, accented, mixed-
script *dictation* on this hardware was the question.

## Setup

- **Engine:** whisper.cpp with the HIP backend on one W7800 (gfx1100), both
  models as ggml f16 (3.1 GB each), `-bs 5 -bo 5 -sow`, `--convert`, same
  container image, same card. The fine-tune runs with `-l he` and no language
  candidates; its model card says language detection and translation were
  degraded in training.
- **Scripted corpus:** 35 read-aloud items (EN/RU/HE, primed glossary terms,
  held-out terms, sound-alike traps), scored with a plain WER (no Hebrew
  normaliser exists anywhere — the score is dominated by script choice for
  embedded Latin names, not by acoustics).
- **Real captures:** 38 Hebrew dictations from the daily capture log, read
  side by side by a Hebrew speaker; no references exist, so the count is
  "clearly better / clearly worse / same".
- Also tried: `ggml-large-v2` (whisper's own Hebrew is reported better on v2
  than v3), the ivrit large-v3-**turbo** fine-tune, the glossary prompt as a
  natural sentence instead of a comma list, and a hosted frontier transcriber
  (Google's, via its native API on five of the clips).

## Results

| pass (all `language=he`, same glossary prompt) | scripted corpus, 10 HE items, WER | real captures vs stock |
|---|---:|---|
| stock large-v3 | 45.9 % | — |
| **ivrit-ai large-v3** | 44.9 % | **better 12 · worse 4 · same 22** |
| ivrit-ai large-v3-turbo | 53.1 % | worse |
| stock large-v2 | 52.0 % | worse |
| stock large-v3, glossary as a sentence | 44.9 % | not run |
| hosted frontier transcriber (5 clips) | — | tie overall: 1 better, 1 worse (a Devanagari-script hallucination mid-sentence), 2 ties, 1 identical; 4.4–5.4 s per clip vs 0.55 s local |

What the fine-tune fixed on real speech: place names (קריית אונו), compound
nouns (לוח אם, אנטנה), time expressions (רבע שעה), and — consistently — no
hallucinated speaker labels ("שי:", "שניים:") that stock large-v3 prepends to
Hebrew utterances. What it broke: nothing on Hebrew audio, but see below.

## Two failure modes that shaped the deployment

1. **It is Hebrew-only by construction.** English or Russian spoken into it
   comes back as Hebrew-letter garbage (its language ID is degraded on
   purpose). The client therefore routes to the lane only on a signal it
   trusts — the keyboard layout of the window being dictated into — never on
   auto-detect, and stock large-v3 stays the default for everything else.
2. **A one-word clip came back empty** ("תודה" → `""`) where stock large-v3
   transcribed it. The client retries the default lane on an empty reply.

## What the scripted corpus taught instead

Its Hebrew score is ~45 % for every model because the references write
embedded product names in Latin (`Grafana`, `Tailscale`, `viewmodel`) and
every engine writes them phonetically in Hebrew (גרפנה, טייל סקייל). Counting
genuine mishearings by hand gives ~15 % for both models — the rest is script
choice, which no acoustic model decides. The fix for that lives in the LLM
cleanup stage after STT: with the glossary in its prompt, Latin retention on
those items went from 6/18 to 10/18 — but only when the rule carried concrete
examples; stated abstractly, the same model Latinised the *people's* names and
turned אריה ("a lion at the zoo") into a listed name. The corpus's trap items
exist precisely for that.

## Takeaways

- A 10-item, ~100-word eval cannot rank Hebrew models: the 95 % interval is
  roughly ±8 points. Real captures read by a speaker settled in an afternoon
  what the corpus could not.
- Fine-tunes that trade language ID for accuracy need a routing signal outside
  the audio. Layout routing is that signal for dictation.
- For mixed-script technical Hebrew, the visible defect is transliteration, not
  mishearing — measure and fix it in text, after STT.
- Nothing hosted was worth the wire: the frontier API tied on quality, cost
  3× the latency, and on its free tier trains on the audio.
