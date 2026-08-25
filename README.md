# rdna3-llm-bench

Coding-LLM serving benchmarks and multi-GPU engineering findings from a
**dual AMD Radeon Pro W7800 (2× 48 GB, gfx1100 / RDNA3)** workstation.

Public data for LLM serving on RDNA3 — especially *dual-GPU* RDNA3 — is close
to nonexistent: vendor numbers cover CDNA datacenter parts, and consumer-card
reports are single-GPU anecdotes. This repo publishes what one heavily-used
box measured: model bake-offs on a fixed harness, tensor-parallel debugging
that landed two upstream ROCm issues and a working TP=2 recipe, and the first
independent field data (to our knowledge) for vLLM's native RDNA3 W4A16
kernels.

Everything here was measured on the machine described below. n=1 hardware,
honest methodology, failure modes documented. gfx1100 findings should
transfer to RX 7900 XTX/XT-class cards (same silicon family), with less VRAM.

## Hardware

| Component | Value |
|---|---|
| GPUs | 2× AMD Radeon Pro W7800 32CU (48 GB GDDR6 ECC each, gfx1100) |
| PCIe | both cards CPU-attached, PCIe Gen4 **x8/x8** (dual root ports — no P2P bridge between them) |
| Platform | ASUS ProArt B850-Creator WiFi (AM5), DDR5-6400, FCLK pinned 2000 MHz |
| Power | 220 W cap per GPU (vBIOS ceiling 230 W) |
| OS | Ubuntu 24.04 LTS |
| Engines | Ollama (pinned `0.32.9-rocm`), vLLM ROCm images (`0.19.1`/rocm 7.13 and `0.23.0`/rocm 7.14 `rdna`), llama.cpp |

## Headline results (2026-08 unified final)

Nine candidates, 20B–123B, one engine version (Ollama 0.32.5-rocm, GGUF
q4_K_M, layer-split over the 96 GB pool), one corrected harness.
HumanEval-100 (official harness, sandboxed) + 12 synthetic tasks + a
multi-turn agentic tool loop. Full tables and caveats:
[`results/2026-08-unified-final.md`](results/2026-08-unified-final.md).

| candidate | pass@1 | decode tok/s | agentic tool loop |
|---|---:|---:|---|
| **qwen3-coder:30b** (30B-A3B) | **0.98** | **93.6** | **converged, 4 turns** |
| gemma4:26b-a4b | 0.96 | 70.5 | 12-turn cap |
| qwen3-coder-next (80B-A3B) | 0.96 | 67.1 | 12-turn cap |
| gpt-oss-120b | 0.94 | 74.5 | 12-turn cap |
| glm-4.7-flash | 0.90 | 68.4 | 12-turn cap |
| mistral-small-4-119b | 0.89 | 56.7 | 12-turn cap |
| devstral-2-123b | 0.88 | 7.0 | converged, 7 turns |
| devstral-small-2 | 0.88 | 31.5 | 12-turn cap |
| glm-4.5-air | 0.63¹ | 36.2 | 12-turn cap |

¹ thinking-budget-bound even at a 3072-token cap — a floor, not a capability
measurement. See the [thinking-models finding](findings/thinking-models-vs-harnesses.md).

**2026-08-13 addendum:** Meta's **Muse Glimmer 30B** (dense, thinking, Apache
2.0), benched 3 days post-release: **0.97 pass@1 at 27.5 tok/s** — best dense
score we've measured, but 3.5× slower than the equal-quality 3B-active MoE and
non-converging on the tool loop. Details:
[`results/2026-08-muse-glimmer.md`](results/2026-08-muse-glimmer.md).

The result that matters for agentic use: **small-active MoE wins**. A 3B-active
model beat a 123B dense model on quality *and* ran 13× faster, and was the only
fast model in the field to actually finish a multi-turn tool loop. On
bandwidth-bound consumer hardware, "biggest model that fits" is the wrong
question.

## Engineering findings

- **[Dual-RDNA3 tensor parallelism](findings/rccl-dual-gpu-tp2.md)** — vLLM
  TP=2 on two root-port-separated RDNA3 cards hits *two* independent RCCL
  defects: silent all_gather corruption (root-caused to a lost async H2D copy;
  zero-cost verify-retry patch; [ROCm/ROCm#6565](https://github.com/ROCm/ROCm/issues/6565))
  and a ~7000×-slow P2P transport
  ([ROCm/ROCm#6576](https://github.com/ROCm/ROCm/issues/6576), fixed by
  `NCCL_P2P_DISABLE=1`). With both workarounds: coherent 20.7 tok/s on a 30B
  bf16 model. IOMMU was empirically exonerated.
- **[vLLM native W4A16 kernels on RDNA3](findings/w4a16-native-kernels.md)** —
  vLLM v0.23.0's RDNA3 int4 kernels take the same GPTQ model from 3.9 to
  **29.3 tok/s** single-stream and **125.8 tok/s aggregate at 8-way
  concurrency** (71 ms TTFT). The gate: *symmetric* GPTQ int4 only.
- **[Thinking models break naive harnesses](findings/thinking-models-vs-harnesses.md)** —
  reasoning-mode models emit their chain-of-thought in a separate API field
  and return *empty* content when token-capped. Three successive harness bugs
  took one model from a measured 0.07 to its true 0.96. If your eval doesn't
  capture the reasoning field and budget ≥4k tokens, your reasoning-model
  scores are floors.
- **[Resident models: HIP idles at 80–87 W, Vulkan at 5 W](findings/resident-model-idle-power-vulkan-vs-hip.md)** —
  a llama.cpp server with a dense LLM loaded and no requests draws **80–87 W**
  on the HIP backend ([ROCm/ROCm#2625](https://github.com/ROCm/ROCm/issues/2625))
  and **5 W — the empty-card floor —** on the Vulkan (RADV) backend, same
  model, same card. Ollama residency sits in between (dense ~+40 W, MoE
  ~110 W). If a model must stay warm on RDNA3, pin it under Vulkan.
- Engine version matters more than folklore: an Ollama 0.24 → 0.32.5 upgrade
  alone lifted the winning model's decode **+30%** (75.8 → 93.6 tok/s era-on-era).

## Repo map

- [`methodology.md`](methodology.md) — harness design, scoring, sandboxing,
  normalization, threats to validity
- [`results/2026-08-unified-final.md`](results/2026-08-unified-final.md) — round 2 full data
- [`results/2026-08-muse-glimmer.md`](results/2026-08-muse-glimmer.md) — Muse
  Glimmer 30B addendum (day-3 numbers, same harness)
- [`results/2026-08-per-leg.csv`](results/2026-08-per-leg.csv) — raw per-leg
  latency/throughput (117 rows)
- [`results/2026-05-round1.md`](results/2026-05-round1.md) — round 1 summary
  (single-GPU era; absolute numbers obsoleted by engine upgrades)
- [`findings/`](findings/) — the engineering writeups

## License

Data, results, and prose: [CC-BY-4.0](LICENSE-CC-BY-4.0). Any code that lands
here: [MIT](LICENSE-MIT). Cite as: *rdna3-llm-bench, github.com/BoJl4apa/rdna3-llm-bench*.
