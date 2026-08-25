# Parakeet-TDT-0.6B-v3 vs Whisper large-v3 on 119k words of held-out long-form English

**TL;DR** — On 20 long-form YouTube recordings (118,879 reference words of
uploader-written captions, fully held out), NVIDIA's **Parakeet-TDT-0.6B-v3**
running on **CPU** via [parakeet.cpp](https://github.com/mudler/parakeet.cpp)
beat **whisper large-v3** (whisper.cpp, HIP on a W7800, beam 5) **14.0 % to
17.0 % corpus WER** — 8.2 % to 12.6 % excluding one card whose reference is
suspect for both engines — with equal salient-token recall, near-zero decoder
collapse, and far fewer insertions on typical cards. A 0.6 B CPU model out-
transcribing a 1.5 B GPU model on real-world English is exactly the kind of
result public leaderboards under-sell; per HF's [benchmark-optimization
work](https://huggingface.co/blog/asr-benchmark-optimization), held-out audio
is the only test that counts.

## Setup

- **References:** caption tracks *written by the uploader* (not YouTube ASR) on
  20 videos from a personal knowledge-base corpus — economics, AI, geopolitics
  explainers; 5 min to ~4 h each. Edited captions shift absolute WER up for
  everyone (filler is transcribed but not captioned), so treat the numbers as
  head-to-head, not absolute.
- **Same downloads, same day, same chunking** (5-minute ffmpeg segments, the
  pipeline's anti-collapse measure) for both engines.
- whisper large-v3: whisper.cpp (HIP, gfx1100), `-bs 5 -bo 5 -sow`,
  `entropy_thold 2.8`, translate task. Parakeet: parakeet.cpp `q8_0` GGUF
  (identical WER to f16 in our spot-checks), plain transcription, CPU only
  (Ryzen 9 9900X; ~15× realtime — GPU builds exist via ggml HIP/Vulkan but
  weren't needed).
- Scoring: WER with S/D/I split + salient-token recall (proper nouns and
  numbers surviving into the hypothesis) + a repetition ratio that flags
  whisper's collapse mode.

## Results

| | Parakeet-TDT-0.6B-v3 (CPU, q8_0) | whisper large-v3 (GPU, beam 5) |
|---|---:|---:|
| corpus WER | **14.0 %** | 17.0 % |
| corpus WER excl. the bad-reference card | **8.2 %** | 12.6 % |
| corpus S / D / I | 3457 / 3672 / **9469** | 3618 / 5376 / 11251 |
| salient recall | 82.4 % | 82.9 % |
| cards won | **17 / 20** | 3 / 20 |
| repetition ratio | ~0 on every card | up to 7 % |

Typical-card contrast is starker than the corpus totals (which two pathological
cards dominate for both engines): on an ordinary hour-long explainer Parakeet
lands 2–7 % WER with single-digit insertions, whisper 7–23 % with hundreds.

## The failure-shape argument

For transcription feeding a knowledge pipeline, the S/D/I split matters more
than the aggregate: an LLM cleanup pass can repair *deletions* fairly safely,
but it will polish *insertions* — invented or repeated text — into fluent wrong
claims. Whisper's error mass is insertions (hallucinated sentences on quiet
audio, repetition loops, prompt-bleed); Parakeet under-writes instead: its worst
cards are deletion-heavy, and on true silence it returns an empty string where
whisper produces "Thank you for watching!" or "Subtitles by the Amara.org
community".

Where whisper still wins:

- **Vocabulary priming.** whisper's `prompt` field cut our short-form dictation
  WER by ~5 points on jargon-dense speech; Parakeet has no equivalent, and on a
  proper-noun-heavy dictation corpus whisper+glossary beats it 2.2 % to 7.6 %.
- **Translation and language coverage.** Parakeet transcribes 25 European
  languages and never translates; whisper covers 99 and translates to English.
- **Numbers as digits.** Parakeet spells numbers out ("twenty" vs "20"), which
  conventional WER counts against it.

## Caveats

- n=1 corpus, one accent mix, uploader captions as references (floor-shifted).
- One download failure card excluded for both engines; one card's reference is
  bad enough (I≈7k for both) that we report totals with and without it.
- parakeet.cpp pinned by image digest; whisper.cpp fork at build 7.2.3-era
  (language-candidates patch, upstream PR pending).
