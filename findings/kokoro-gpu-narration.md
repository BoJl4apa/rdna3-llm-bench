# Kokoro on gfx1100: faster synthesis, but resident HIP has an idle-power cost

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
