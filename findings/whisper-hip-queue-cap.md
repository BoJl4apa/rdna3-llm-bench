# Two whisper.cpp HIP servers spin a card at 120 W; `GPU_MAX_HW_QUEUES=1` stops it at no latency cost

**TL;DR** — Two `whisper-server` (whisper.cpp, HIP backend) processes resident on
one Radeon Pro W7800 (gfx1100) hold the card at **120 W / 85 °C / 100 % gfx /
SCLK 2803 MHz** with no requests in flight. Setting
**`GPU_MAX_HW_QUEUES=1`** (HIP's default is 4) on both brings the card to
**4 W**, at the empty-card floor. On a real dictation corpus the cap costs
**nothing measurable**: median and p95 latency agree within one 10 ms step on short
clips and within 1 % on an 88 s file (median 3.338 vs 3.306 s, p95 9.909 vs
9.934 s), the sign of the gap varies, and WER and transcripts are identical.

## Power (measured 2026-09-01, both lanes warm and serving)

| Configuration | Package power | Junction | gfx | SCLK |
|---|---:|---:|---:|---:|
| uncapped, both lanes | **120 W** | 85 °C | 100 % | 2803 MHz |
| uncapped, one lane only | 39 W¹ | — | 0 % | 0 MHz |
| capped (`GPU_MAX_HW_QUEUES=1`), both lanes | **4 W** | 34 °C | 0 % | 0 MHz |

¹ Read with the lane warm and serving. A single resident whisper.cpp HIP lane at rest
reads 13 W on the [idle-power page](resident-model-idle-power-vulkan-vs-hip.md).

The capped figure was read after 10 transcriptions across both lanes. The uncapped
state survives `SIGSTOP` of both processes, so it is a hardware queue left reporting
busy, not a host-side busy-wait. A suspected mechanism is the MES stuck-queue bug
([ROCm/ROCm#2625](https://github.com/ROCm/ROCm/issues/2625)), but neither that report
nor the related kernel-driver issue names this variable. Treat the cap as what was
tested and works, not as a confirmed root cause. It does **not** fix the separate
llama.cpp-HIP resident-LLM spin
([resident-model idle power](resident-model-idle-power-vulkan-vs-hip.md)): 68–72 W
uncapped against 72–76 W capped (gemma-4-12b Q4_K_M, 4096 ctx, 2026-09-01).

## Latency cost (measured 2026-10-11)

`GPU_MAX_HW_QUEUES` is documented by AMD as a debug variable (recommended ≤ 4), and
a cap of 1 is off the beaten path, so the question is what it costs. whisper-server
serialises every request behind one process-wide mutex, so the cap has no
request concurrency to lose. The open question was whether beam search
(`-bs 5 -bo 5`) pays a per-launch queue penalty on long-form speech.

**Method.** Throwaway `whisper-server` containers on one card, **one arm at a time**,
with the production lanes untouched. Both arms used the same image, devices,
model (ggml large-v3) and command line, VAD included, and differed only in the
variable: arm A `GPU_MAX_HW_QUEUES=4` (the HIP default), arm B `=1`. Each arm got
one warm-up, then the 35 clips of a recorded mixed-language dictation corpus
(English, Russian, Hebrew), then an 87.6 s English recording posted 20 times.
WER is scored against the corpus references.

| | cap 1 | HIP default 4 |
|---|---|---|
| 35 clips, median / p95 latency | 0.59 / 0.77 s | 0.60 / 0.76 s |
| 88 s file ×20, median / p95 | 3.338 / 9.909 s | 3.306 / 9.934 s |
| WER (per language and stratum) | identical | identical |

Clip latencies are logged at 10 ms resolution. p95 is linear-interpolated, and the
verdict holds under nearest-rank and exclusive p95 as well: the clip gaps are at most
one 10 ms step, the long-file gaps are at most 0.033 s (1 %), and the sign of the gap varies. The long-file outliers (5–12 s) fall on the same run indices in both
arms, with identical output lengths, so they come from decoding that recording, not
from the cap.

**Settle after load** (observed, not logged). After each arm's container was removed,
the card drew 21–46 W at 0 % use for 90–120 s before dropping to 4 W. That happened in
both arms, so it is ordinary DPM ramp-down, not a cap effect.

## Caveats

- n=1 box, one card family (gfx1100), ROCm 7.2.x, a whisper.cpp fork built for HIP.
  Power is `rocm-smi` average package power, not a wall meter.
- One model (large-v3) and one decoding setup (`-bs 5 -bo 5`). A setup that
  issues more concurrent kernel streams per request could behave differently.
- The corpus is 35 short dictation clips plus one long recording. The latency claim
  is "no measurable cost on this workload", not a microbenchmark of queue overhead.
