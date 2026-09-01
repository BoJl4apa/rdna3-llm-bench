# Keeping a model resident on RDNA3: HIP burns 80–87 W idle, Vulkan burns nothing

**TL;DR** — On a Radeon Pro W7800 (gfx1100), a llama.cpp server with a dense
LLM loaded and *zero requests in flight* draws **80–87 W** under the HIP/ROCm
backend and **5 W — the empty-card floor —** under the Vulkan (RADV) backend.
Same model, same box. Ollama (HIP) sits in between: a resident dense model
holds the memory clock up for **~+40 W**, and a resident MoE model spins
shaders at **~110 W / 95 °C**. If you need a model warm on RDNA3 for
latency reasons, put it on the Vulkan build; pinning under HIP is a space
heater.

## Measurements

All readings via `rocm-smi` (`--showpower`, `--showclocks`, `--showuse`), card
otherwise idle, box quiet. Floor for an empty W7800 on this box: **5–6 W,
MCLK 96 MHz, 0 % use.**

| Engine / backend | Model resident | Idle draw | MCLK | GPU use | Note |
|---|---|---:|---:|---:|---|
| (none) | — | 5–6 W | 96 MHz | 0 % | floor |
| whisper.cpp HIP | ggml-large-v3 (3.9 GB) | 13 W | 96 MHz | 0 % | non-LLM; effectively free |
| Ollama 0.32.9-rocm | dense `gemma4:e4b` | ~46 W | **772 MHz** | 0 % | memory clock never drops |
| Ollama 0.32.9-rocm | MoE `gemma4:26b-a4b`, `gpt-oss:120b` | **~110 W** | 96 MHz | **100 %** | shader spin, 95 °C junction, indefinitely |
| llama.cpp `server-rocm` (HIP) | dense 8B Q4 | **80–87 W** | — | spin | from the moment of *load*, no requests |
| llama.cpp `server-vulkan` (RADV) | dense 8B Q4 | 3–5 W | 96 MHz | 0 % | 89 tok/s decode |
| llama.cpp `server-vulkan-b10603` (RADV) | **gemma-4-12b Q4_K_M** (7.1 GB) | **5.0 W** | 96 MHz | 0 % | 63–65 tok/s decode; floor reached 1.5–4 min after the last request or load, then holds |

Post-generation settle under Vulkan (12B): 119 W during decode → 33 W (MCLK
772) → 24 W (MCLK 456) → **5 W (MCLK 96)**, and it stays there. The DPM
step-down is paced, not instant: 90 s on one run, ~4 min on another (with the
other card unloading a model at the same time). A fresh load with no requests
settles the same way.

The HIP behaviour is the long-standing RDNA3 idle-spin bug:
[ROCm/ROCm#2625](https://github.com/ROCm/ROCm/issues/2625),
[ggml-org/llama.cpp#3929](https://github.com/ggml-org/llama.cpp/issues/3929).
The Ollama MoE number is the same class of bug hitting a different kernel; the
Ollama dense number is a separate, milder effect (DPM never returns the memory
clock to its floor while the allocation is live).

## Why it matters

For a chat/coding server that is busy, none of this matters — the card is
working anyway. It matters for **latency-sensitive, low-duty consumers**:
dictation cleanup, autocomplete, a classifier behind a hotkey. There the
choice was between a cold-load hit on every idle gap (we measured 4.5–5.3 s
cold vs 0.5–0.8 s warm on roughly half of real dictations) and ~40–110 W of
standing draw to avoid it. Vulkan removes the trade: the model stays loaded,
every call is warm, and the card idles at its floor.

## Recipe

Container image: `ghcr.io/ggml-org/llama.cpp:server-vulkan-<build>` (pin the
build; `server-vulkan` floats). It needs only the DRM render nodes, not
`/dev/kfd`:

```
docker run -d --device /dev/dri --group-add <video-gid> --group-add <render-gid> \
  -v /models/dictate:/models/dictate:ro -p 127.0.0.1:8091:8091 \
  ghcr.io/ggml-org/llama.cpp:server-vulkan-b10603 \
  -m /models/dictate/gemma-4-12b-it-Q4_K_M.gguf --alias dictate \
  --device Vulkan2 -ngl 99 --ctx-size 4096 --parallel 1 --reasoning off \
  --host 0.0.0.0 --port 8091
```

Landmines:

- **RADV enumerates the CPU's iGPU as a Vulkan device.** On a Ryzen 9000 box
  `--list-devices` shows `Vulkan0` = the Raphael iGPU (reporting *system RAM*
  as its VRAM), `Vulkan1`/`Vulkan2` = the two W7800s. Without `--device`
  llama.cpp will happily split layers across all three. Select explicitly.
- **Use a standalone GGUF.** Ollama's `gemma4:12b` blob is a multimodal bundle
  (text + CLIP projector) that llama.cpp does not load standalone; fetch the
  text-only GGUF from Hugging Face instead.
- gemma-4 is a thinking model; `--reasoning off` at the server means no
  client needs to pass a `think:false` equivalent.
- Decode under Vulkan: 63–65 tok/s on the 12B Q4_K_M, 89 tok/s on an 8B. We
  did not A/B decode speed against the HIP build — this lane exists for its
  idle draw, not its throughput. A one-sentence cleanup is ~0.5 s end-to-end.

## Caveats

- n=1 box, one card family (gfx1100). Numbers from `rocm-smi` average package
  power, not a wall meter.
- The 8B HIP/Vulkan pair was measured 2026-08-24; the 12B Vulkan lane
  2026-08-25 (llama.cpp build 10603). ROCm 7.x host, Ubuntu 24.04.
- Tested 2026-09-02: **the stuck-queue spin costs power only.** With a
  whisper.cpp/HIP container deliberately uncapped and its card pinned at
  100% busy / 78–104 W, an LLM serving 20 tool-loop runs on the *other* card
  measured identical convergence (1/20 vs 1/20 on the quiet control) and
  ≤1% decode delta (93.9 vs 94.7 tok/s). Details: [tool-loop convergence
  instability](tool-loop-convergence-instability.md).
- Not tested: whether newer ROCm releases fix the HIP idle spin. #2625 has
  been open since 2023.
