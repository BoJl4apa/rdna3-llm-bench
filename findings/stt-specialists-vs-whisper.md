# Four STT specialists vs Whisper on mixed-language dictation (2026-10)

**TL;DR** — None of four recent speech-recognition specialists —
**GigaAM-v3** (Russian), **Qwen3-ASR-1.7B** (English and Russian),
**Caspi-1.7B** (Hebrew) and **Granite Speech 4.1 2B** (English) — beat the
lanes they would replace on our dictation audio: stock **whisper large-v3**
for English and Russian, and **ivrit.ai's large-v3 fine-tune** for Hebrew
(see the [Hebrew-lane finding](hebrew-stt-lane.md)). We set three criteria
before the run. Every candidate fails every criterion that applies to it.
The decisive mechanism is **code-switching**. A per-language lane reached by
*detected* language also receives the mixed clips: Russian or Hebrew speech
with English product names inside. The specialists destroy those names.
GigaAM and Qwen3-ASR kept **0 of 14** embedded English tokens in Russian
speech, and Caspi kept **0 of 16** in Hebrew. Whisper kept **13 of 14**. The
serving notes below are the part most likely to be useful to someone else
running these models on RDNA3.

## The question and the criteria

A candidate earns a lane only if it meets all three criteria, fixed before
any audio was run:

1. **Corpus WER at least 2 points better** than the live lane for its
   language.
2. **More real clips better than worse** in a blind A/B against the live
   lane.
3. **Embedded Latin-script tokens survive** on the code-switched corpus
   clips, at least as well as on the live lane. This applies to the Russian
   and Hebrew candidates only.

## Setup

- **Box and date:** this box, 2026-10-01. Every candidate ran on one W7800 or
  on the CPU, and the live lanes kept the other card.
- **Scripted corpus:** 35 read-aloud items: 12 English, 13 Russian and 10
  Hebrew. The items are glossary-primed terms, held-out terms, sound-alike
  traps and regressions, all under 11 s. The scorer computes word-level edit
  distance after NFKC normalisation, lowercasing and stripping punctuation. It
  is a different scorer and a different date from the
  [Hebrew-lane page](hebrew-stt-lane.md), so its numbers are not comparable
  digit for digit. Five Russian items and six Hebrew items carry English
  product names (Grafana, Prometheus, Redis, DataGrid and the like): 14 Latin
  tokens in Russian and 16 in Hebrew.
- **Real captures:** 50 everyday push-to-talk dictations, plus the 38 real
  Hebrew dictations from the Hebrew-lane page. They have no references. Each
  candidate's transcript was paired with the live lane's for the same clip.
  That gave 136 pairs, presented in an order randomised with seed 74, with
  the key withheld from the judge. One judge read both texts for every pair
  and marked A, B or same, **without access to the audio**.
- **Request fields** were the same for every pass. The deployed glossary went
  as `prompt` wherever the engine accepts one. Where it doesn't, a
  context-carrying variant was run as a separate pass (see Serving).

## Results

| candidate (best variant) | lang | corpus WER: candidate / live | ① | real clips: better / worse / same | ② | Latin tokens kept, mixed corpus clips: candidate / live | ③ |
|---|---|---:|:---:|---|:---:|---|:---:|
| Granite Speech 4.1 2B, glossary in the instruction | EN | 17.2 % / 18.9 % (12 items) | ✗ (1.7) | 13 / 21 / 6 | ✗ | — | — |
| Qwen3-ASR-1.7B, glossary as system context | EN | 21.1 % / 18.9 % | ✗ | 7 / 10 / 23 | ✗ | — | — |
| Qwen3-ASR-1.7B, glossary as system context | RU | 17.9 % / 11.1 % (13 items) | ✗ | 1 / 2 / 3 | ✗ | 4 of 14 (1 of 5 clips) / 13 of 14 (4 of 5) | ✗ |
| GigaAM-v3 e2e-RNNT | RU | 27.4 % / 11.1 % | ✗ | 3 / 3 / 6 | ✗ (tie) | 0 of 14 (0 of 5) / 13 of 14 (4 of 5) | ✗ |
| Caspi-1.7B | HE | 54.1 % / 43.9 % (10 items) | ✗ | 5 / 26 / 7 | ✗ | 0 of 16 (0 of 6) / 4 of 16 (1 of 6) | ✗ |

The real-clip column counts the pairs per candidate: 40 English, 38 Hebrew,
and 12 Russian for GigaAM versus 6 for Qwen3-ASR. The two Russian candidates
were fed slightly different Russian subsets of the 50, so their counts are
not comparable with each other, and at n=6 and n=12 they are close to
anecdote.

Other variants were scored and none changes a verdict:

| variant | lang | corpus WER |
|---|---|---:|
| Qwen3-ASR, plain transcription endpoint (glossary does not reach the model) | EN / RU | 22.8 % / 25.6 % |
| Qwen3-ASR, plain endpoint, Latin tokens kept in Russian | RU | 0 of 14 |
| Granite, no glossary | EN | 24.4 % |
| Caspi, glossary as system context | HE | 58.2 % |
| Parakeet-TDT-0.6B-v3 (CPU) | EN | 22.2 % |

Parakeet's 22.2 % against whisper's 18.9 % on short, glossary-heavy
dictation matches the vocabulary-priming caveat in the
[long-form Parakeet finding](parakeet-vs-whisper-long-form.md). Parakeet
wins on long-form English and loses where a prompt carries the jargon.

## What decides it: code-switching

A dictation client that picks a lane by *detected* language sends every clip
the detector calls Russian to the Russian lane, and that includes Russian
with English product names inside. On those clips:

- **GigaAM-v3** and plain **Qwen3-ASR** transliterated every English name
  into Cyrillic: Vaultwarden became «Волтварден», for example. Qwen3-ASR with
  the glossary as context recovered 4 of the 14 names. Whisper large-v3 kept
  13 of 14.
- **Caspi** kept none of the 16 English names inside Hebrew speech. The live
  ivrit.ai lane is weak here as well (4 of 16), which is the transliteration
  problem the [Hebrew-lane page](hebrew-stt-lane.md) already describes. On the
  real Hebrew captures the judge flagged 7 Caspi pairs as code-switched. The
  live lane alone kept the Latin name on 4, Caspi alone on 2, and both on 1.

For a monolingual lane, these clips would be out of scope. For a lane that a
language detector feeds, they are the normal input.

## Failure modes worth knowing

- **Caspi ignores the language field.** It tags every output as Hebrew, and
  vLLM's `to_language` has no effect. English audio came back as a
  Hebrew-English mix, and Russian audio came back as Hebrew-script gibberish.
  It is only safe behind a routing signal that does not come from the audio,
  like the ivrit.ai lane. Two identical runs over the 38 real Hebrew clips
  gave 35 of 38 identical texts.
- **Qwen3-ASR's context slot echoes on silence.** With the glossary as system
  context, it returned the glossary itself as the transcript on 5 near-silent
  clips. With no language forced, it emits `嗯。` on near-silence. A Hebrew
  clip, a language it does not cover, came back in Persian script.
- **Caspi with context** fell into repetition loops on 2 clips (~1,500–1,850
  characters, ~8.5 s each). The plain pass did not.
- **Granite's output** is lowercase and mostly unpunctuated, which our scorer
  does not penalise. On non-English speech it produces fake German or Czech.
  With the glossary in its instruction, it emits filler words ("okay", "um")
  on near-silent clips where the plain pass returns empty.

## Serving on RDNA3 (gfx1100)

| model | engine | device | memory | cold start | median s/clip |
|---|---|---|---|---:|---:|
| whisper large-v3 (live) | whisper.cpp HIP, beam 5 | GPU | ~4.4 GB VRAM | — | 0.77 |
| ivrit.ai large-v3 (live) | whisper.cpp HIP, beam 5 | GPU | ~4.4 GB VRAM | — | 0.61 |
| Parakeet-TDT-0.6B-v3 (live) | parakeet.cpp, q8_0 | CPU | 8.4 GiB RSS | — | 0.24 |
| GigaAM-v3 e2e-RNNT | HF remote code, PyTorch 2.5.1, fp32 | CPU, 8 threads | 1.5 GiB RAM | 0.56 s (weights only) | 0.11 |
| Granite Speech 4.1 2B | llama.cpp `server-rocm`, Q8_0 GGUF + f16 mmproj | GPU | 4.36 GB VRAM | 5.6 s | 0.20 |
| Qwen3-ASR-1.7B | vLLM 0.19.1 ROCm 7.13 | GPU | ~12 GB at util 0.25 (3.99 GiB weights) | 96 s | 0.13–0.15 |
| Caspi-1.7B | vLLM 0.19.1 ROCm 7.13 | GPU | ~11.7 GB at util 0.25 | 99 s | 0.15 |

The latency column is not a fair race. The live Whisper lanes run beam 5,
behind an authenticating proxy, on the card that serves live traffic. The
candidates ran at temperature 0 (GigaAM: greedy RNNT) on a quiet card.

- **vLLM 0.19.1 ROCm** (`rocm/vllm:rocm7.13.0_gfx110X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`)
  serves Qwen3-ASR and Caspi (`Qwen3ASRForConditionalGeneration`) on gfx1100
  once `librosa` and `soundfile` are pip-installed into the image.
- **Engine crash:** the engine died three times across the run (two crash
  logs kept), all with `IndexError: index is out of bounds for dimension with
  size 0` in the audio tower (`qwen3_omni_moe_thinker.py`, `feature_lens`
  arriving as 0). That raises `EngineDeadError`, and every later request
  returns 500. It is non-deterministic, and the same clips pass on retry.
  `--no-async-scheduling` alone did not stop it. With
  `AMD_SERIALIZE_COPY=3` added, the Caspi server took 152 logged requests
  with no crash. That is a mitigation on a small n, not a root cause.
- **On vLLM's OpenAI transcription endpoint, Qwen3-ASR ignores `prompt` and
  `language`.** `language` is validated and then dropped: `language=en` on
  Russian audio returns the same Russian text. The glossary on and off gives
  byte-identical output. Only vLLM's own `to_language` field forces the
  language. A glossary reaches the model only as a **system message on
  `/v1/chat/completions`**. That route needs a custom chat template, because
  the shipped one drops assistant turns. The template sends an assistant
  prefix `language <Name><asr_text>` with `continue_final_message=true`. That
  route cut Russian corpus WER from 25.6 % to 17.9 %.
- **Granite Speech 4.1** runs on llama.cpp's `server-rocm` image on one card
  through `/v1/chat/completions` with `input_audio`. The log warns that 16
  unary ops in the audio encoder fall back to CPU, and it labels audio input
  experimental. The build also exposes `/v1/audio/transcriptions`, which adds
  its own hidden instruction. We did not use it.
- **GigaAM-v3** has no prompt or language input. It ran on CPU from the
  model's HF remote code under transformers 4.46.3. It needed an empty
  `pyannote` stub to pass transformers' import check, and `.float()` because
  the weights are fp16.

**Licences:** Qwen3-ASR-1.7B is Apache-2.0 and GigaAM-v3 is MIT. Caspi-1.7B
is **CC-BY-NC-4.0**, which rules it out of any commercial use whatever its
accuracy. Granite Speech's licence was not checked against the GGUF repack we
ran.

## Caveats

- **The corpus is small:** 12, 13 and 10 items, 35 in total and about 100
  words per language. A 1.7-point gap on 180 English words is inside the
  noise, so criterion ① is a floor, not a fine ranking.
- **The clips are short:** every clip is under 11 s, and there is no
  long-form audio. Nothing here says how these models do on meetings or
  lectures.
- **There is a single speaker,** with one accent per language.
- **The judge had no audio.** Real-clip verdicts are judgements of which text
  is more plausible, made from the transcripts alone. Where both texts were
  plausible, the pair was marked "same", so the ties hide real differences.
- **One run per pass,** except Caspi's determinism check. There is no
  run-to-run variance for the others.
- **This is our audio.** The candidates are strong on their published
  monolingual benchmarks. This page measures a code-switched dictation
  workload, which those benchmarks do not cover.
