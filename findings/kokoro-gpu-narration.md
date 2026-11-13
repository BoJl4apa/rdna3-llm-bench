# Kokoro on gfx1100: resolving idle spin with GPU placement and queue budgets

**Update:** placement and queue limits resolved the measured idle spin without
a kernel upgrade. Warm production audio and coding/RAG GPUs both reach 0% busy;
see the final qualification below. Earlier sections preserve the investigation.

Measured 2026-11-13 on one Radeon Pro W7800 48 GB (gfx1100), Ryzen 9 9900X,
Ubuntu 24.04. The GPU already held two whisper.cpp STT servers and a
llama.cpp Vulkan cleanup server. This is a bounded deployment comparison,
not a general ranking or a pure CPU/GPU microbenchmark.

## Kokoro latency

`af_bella`, speed 1.0, synthetic English, 24 kHz mono PCM over bridge-local
HTTP. First-audio timing ends after reading 4096 PCM bytes; HTTP headers
alone do not count. Complete timing includes the entire response body.
Three repetitions; startup model loading is excluded.

| Configuration | Short first audio | Paragraph first audio | Paragraph complete |
|---|---:|---:|---:|
| Kokoro-FastAPI CPU v0.4.0 | 0.81–0.96 s | 2.07–2.21 s | 3.70–4.28 s |
| Kokoro-FastAPI ROCm v0.10.0 | 0.18–0.24 s | 0.31–0.32 s | 0.60–0.61 s |

The short output lasts about six seconds, the paragraph about 25.5 seconds.
CPU and GPU use different wrapper versions; do not attribute the entire
gain to hardware alone. GPU v0.4.0 took 0.97 s for the first short text and
2.12 s for the first paragraph, falling to 0.13 s and 0.48 s on repetition.

The newer wrapper disables MIOpen on ROCm to avoid per-shape kernel
compilation. Tested GPU environment: `DEVICE_TYPE=cuda`,
`ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`,
`HSA_OVERRIDE_GFX_VERSION=11.0.0`, `GPU_MAX_HW_QUEUES=1`,
`TORCH_BLAS_PREFER_HIPBLASLT=0`, `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`,
`ENABLE_MIOPEN=false`. The image was
`ghcr.io/remsky/kokoro-fastapi-rocm:v0.10.0` at digest
`sha256:53ef42fec42f173aab83d9750a7cad3ec899e8050c61b1e5ff6e670004fa93cc`.

During simultaneous generation of a 76-second narration, a synthetic STT
fixture remained correct at 0.69–1.12 s versus 0.58–0.65 s without TTS.
Cleanup remained correct at 0.41–0.55 s versus 0.33–0.41 s. Three runs;
this is a coexistence check, not a load-capacity claim.

## Residency is not power-free

Kokoro added roughly 2.6 GiB resident GPU memory. With no requests in flight,
the card stayed at 76–83 W and 100% reported activity. Stopping the Kokoro
process returned it to 17–18 W / 0%. These are `rocm-smi` package-power
readings, not wall-meter measurements, and include the existing services.

Neither default queue allocation nor `GPU_MAX_HW_QUEUES=1` resolved it.
`AMD_SERIALIZE_KERNEL=3` plus `AMD_SERIALIZE_COPY=3` also failed and slowed
paragraph generation to about 1.48 s. No kernel, firmware or host scheduling
changes were tried. The similarity to other gfx1100 HIP idle findings is
not proof of an identical cause. See the existing
[resident model power comparison](resident-model-idle-power-vulkan-vs-hip.md)
and [Whisper queue-cap finding](whisper-hip-queue-cap.md).

## Small RU/EN GPU audition

Separate disposable environment: PyTorch 2.10 / ROCm 7.13, BF16 and SDPA
for Qwen, no FlashAttention. Complete-generation timings, not streaming
latency. Peak allocated VRAM is PyTorch's counter, not total process VRAM.

| Model | Synthetic workload | Generation | Peak allocated VRAM |
|---|---|---:|---:|
| Qwen3-TTS 0.6B CustomVoice, Ryan | EN, 8–8.7 s output, n=2 | 5.90–6.25 s | up to 2.64 GiB across EN/RU |
| Same Qwen model | RU, 8.1–11 s output, n=2 | 5.65–7.84 s | same |
| Silero v5_6_ru, xenia, 48 kHz | Four fresh RU texts, 5–6 s output | 0.62 s first; 0.010–0.011 s thereafter | about 0.42 GiB |

Silero timing includes `apply_tts` and GPU synchronization, before file
encoding. The fresh texts guard against a repeat-input cache explanation.
Whisper transcription of five Silero clips matched the words except for
one number-format difference (ten versus 10). Qwen EN matched; Qwen RU had
one singular/plural difference. This is an ASR sanity check, **not human
voice-quality acceptance** or a general accuracy benchmark.

The cached evaluation image's AMD SMI enumeration reported zero devices
while HIP reported one. A probe-only fallback to HIP enumeration was used;
this environment is not qualified as a deployment image. Qwen is Apache-2.0;
the tested Silero Russian weights are noncommercial, distinct from its MIT
CIS-base models. No Russian service deployment follows from this audition.

Sources: [Kokoro ROCm guidance](https://github.com/remsky/Kokoro-FastAPI/blob/v0.10.0/docs/troubleshooting.md),
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
[Silero](https://github.com/snakers4/silero-models),
[related gfx1100 pending-barrier report](https://github.com/ROCm/TheRock/issues/9484).

## Follow-up: placement changes the idle result (2026-11-13)

The high idle draw is **not universal to resident Kokoro**. A second physical
W7800, with no Ollama model loaded, reached 19 W / 0% busy while keeping Kokoro
warm. Same image and synthetic fixtures, `GPU_MAX_HW_QUEUES=2`, one repetition:

| Placement | Idle at 15/30/45 seconds | Short first audio | Paragraph first / complete |
|---|---|---|---|
| Shared audio GPU | 74/75/76 W, 100% busy | 0.205 s | 0.313 / 0.602 s |
| Second GPU, no Ollama model loaded | 24/19/19 W, 0% busy | 0.209 s | 0.316 / 0.607 s |

Both cards have eight KFD compute slots. After the probes exited, the audio
GPU had seven compute queues: two per Whisper process and three from a HIP
reranker. The reranker also had three queues on the second card; that card
was not empty of all other driver clients. Queue oversubscription is the
leading hypothesis, not a proven causal trace. This does not qualify sharing
Kokoro with loaded Ollama. The next comparison should constrain the reranker
away from the audio GPU, then repeat synthesis, STT and idle measurements.

Production temporarily returned to the CPU rollback rather than retaining
the high standing draw. Narration still passes a real client-path WAV check.
The host remains on kernel 6.8.0-149 with MES 0x80; no host tuning changed.

There is a relevant newer [AMD GFX11 oversubscription workaround for Linux
7.4](https://github.com/ROCm/TheRock/issues/9141#issuecomment-6211588196),
confirmed present in v7.4-rc2 source. On this date that release is still RC.
A [distinct pending-barrier reproducer remains unresolved with newer firmware
and userspace](https://github.com/ROCm/TheRock/issues/9484#issuecomment-6729369200),
so a generic ROCm upgrade is not a demonstrated cure. Investigate queue
placement before a disruptive kernel experiment.

## Resolution: budget compute queues as well as VRAM

The audio card had seven compute queues before Kokoro: two per whisper.cpp
process, plus three from a reranker that initialized both GPUs. Confining that
reranker to GPU1 removed its GPU0 queues. With Kokoro resident, GPU0 now uses
six queues and idles normally. The unchanged reranker image and synthetic
ranking fixtures returned identical relevance scores before and after placement.

The other card needed a limit too: coder + embedding under default Ollama HIP
queue allocation used eight compute queues, plus three from rerank = eleven.
That recreated 73–74 W / 100% idle activity on GPU1. Setting Ollama
`GPU_MAX_HW_QUEUES=1` reduced the two tested runners to four compute queues.
`OLLAMA_MAX_LOADED_MODELS=2` keeps their total with rerank at seven, below the
eight slots exposed by this kernel. The cap is per process; runtime-internal
queues mean it is not the actual KFD queue count. Measure new workloads.

| Warm configuration after concurrent inference | GPU0 idle | GPU1 idle |
|---|---|---|
| Reranker moved; Ollama uncapped | 15 W / 0% | 73–74 W / 100% |
| Queue-capped candidate | 14–15 W / 0% | 18–25 W / 0% |
| Deployed production recheck | 12–18 W / 0% | 16–24 W / 0% |

Samples were taken 15, 30 and 45 seconds after requests, while models remained
resident. These are card telemetry readings, not whole-host power. Audio used
about 18.3 GiB VRAM; coding/RAG about 28.8 GiB. No kernel, firmware or BIOS change.

Three simultaneous synthetic rounds covered STT, Vulkan cleanup, ~76 seconds
of Kokoro narration, coding, embedding and ranking. All functional assertions
passed. Warm short-fixture coding measured 94.9–101.3 tok/s with the cap versus
86.7–102.5 without; this is a 16-token smoke check, not a general throughput
benchmark. STT took 0.749–0.752 s in the capped candidate versus 0.624–0.672 s
uncapped; narration completed in 2.133–2.173 s versus 2.145–2.244 s. Retain
these limits rather than claiming statistical equivalence.

Coder used its configured 65,536-token context and embedding reported 40,960;
both were entirely GPU resident. The test did not fill the context windows.
A third model request completed and evicted an idle model; choices remain
available but rotating among more than two pays reload latency. Parallelism
one, five-minute expiry, Flash Attention and q8 KV cache were retained.

Production Kokoro is back on GPU. Real client-path streaming delivered short
first audio in 0.183–0.232 s and paragraph first audio in 0.329–0.349 s; valid
24 kHz mono WAV also passed. Reconciliation was idempotent on a second run.

For additional languages, free VRAM does not justify independent unbounded
HIP workers: prefer a shared inference process/queue budget and remeasure
residency, concurrency and idle power. Other model pairs, build contention
and reliable idle-power monitoring remain separate qualification work.
