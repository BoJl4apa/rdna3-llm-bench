# 2026-08 bake-off — unified final

Nine candidates, 20B–123B parameters, benched 2026-08-04/05 on the dual-W7800
box. One engine (`ollama/ollama:0.32.5-rocm`, GGUF q4_K_M, layer-split over
96 GB), one corrected harness (reasoning capture, best-block extraction,
HumanEval cap 3072 / synth cap 4096). See [`../methodology.md`](../methodology.md).

## Final table

| candidate | params | pass@1 | decode tok/s | agentic tool loop |
|---|---|---:|---:|---|
| **qwen3-coder:30b** | 30B-A3B MoE | **0.98** | **93.6** | **converged, 4 turns** |
| gemma4:26b-a4b | 26B-A4B MoE | 0.96 | 70.5 | 12-turn cap |
| qwen3-coder-next | 80B-A3B MoE | 0.96 | 67.1 | 12-turn cap² |
| gpt-oss-120b | 120B MoE | 0.94 | 74.5 | 12-turn cap |
| glm-4.7-flash | — | 0.90 | 68.4 | 12-turn cap |
| mistral-small-4-119b | 119B | 0.89 | 56.7 | 12-turn cap |
| devstral-2-123b | 123B dense | 0.88 | 7.0 | converged, 7 turns |
| devstral-small-2 | — | 0.88 | 31.5 | 12-turn cap |
| glm-4.5-air | 106B-A12B MoE | 0.63¹ | 36.2 | 12-turn cap |

¹ Still thinking-budget-bound at 3072 (32 empty completions; scores 62/63 on
what it finishes). It reasons longest of the field; measuring it properly
needs 8–12k-token budgets, impractical at 36 tok/s. Read as a floor.
² Converged at 5 turns in an earlier (512-cap) harness generation; the cap
change altered its loop behavior — noted, not investigated.

**"12-turn cap"** means the model never returned a turn free of tool calls
within 12 turns — it never declared itself done. All nine candidates used
native structured tool calls; none degraded to text-described tool use.

## Reading the table

- **Small-active MoE dominates for agentic serving.** The 3B-active winner
  beat the 123B dense model on pass@1 (0.98 vs 0.88) at **13× the speed**, and
  fast tool-loop convergence was unique to it. Decode on this hardware is
  VRAM-bandwidth-bound: active-parameter bytes, not total parameters, set the
  speed ceiling.
- **Convergence separates more than pass@1.** At q4 on this class of model,
  HumanEval is nearly saturated (0.88–0.98). Whether a model can *finish* a
  multi-turn tool task cleanly is the differentiator that matters in an
  agentic harness, and only 3 of 9 managed it.
- **Speed is a quality feature in tool loops:** a 4-turn convergence at 93
  tok/s is interactive; the same convergence at 7 tok/s (devstral-2) is not.

## Per-leg latency/throughput

Raw per-leg data (117 rows: 9 candidates × [HumanEval-100 + 12 synth legs]) in
[`2026-08-per-leg.csv`](2026-08-per-leg.csv): TTFT, total latency, completion
tokens, decode tok/s per leg. Notes:

- Headline tok/s above are cross-leg aggregates; per-leg values vary ±2%.
- TTFT on thinking models includes their reasoning phase (e.g. gemma4's
  13.2 s mean HumanEval "TTFT" is mostly chain-of-thought, not prefill).
- devstral-2-123b's ~7 tok/s is what 123B dense at q4 looks like on layer-split
  GDDR6 — physics, not a bug.

## Superseded generations (kept for the record)

The first scoring pass (512-token cap, no reasoning capture) produced this
now-void all-round lane — preserved because the *delta* is the finding:

| candidate | gen-1 pass@1 | final pass@1 | cause |
|---|---:|---:|---|
| gemma4:26b-a4b | 0.07 | 0.96 | 77/100 empty: whole budget spent thinking |
| glm-4.7-flash | 0.13 | 0.90 | 69/100 empty, same |
| glm-4.5-air | 0.25 | 0.63 (floor) | 57/100 empty, same |
| gpt-oss-120b | 0.35 | 0.94 | 94/100 capped mid-docstring (verbosity), +16 lost to fence-extraction bug |

Anatomy in [`../findings/thinking-models-vs-harnesses.md`](../findings/thinking-models-vs-harnesses.md).

## Engine matrix (same box, same period)

| Lane | Status | Numbers |
|---|---|---|
| Ollama / llama.cpp layer-split (96 GB pool) | production | winner 93.6 tok/s; 119B MoE ~57; 106B MoE ~36; 123B dense 7 |
| vLLM TP=1 W4A16 (v0.23.0 native RDNA3 kernels) | proven | 29.3 tok/s single, 125.8 tok/s aggregate @ 8-way, 71 ms TTFT ([details](../findings/w4a16-native-kernels.md)) |
| vLLM TP=2 bf16 (patched RCCL + `NCCL_P2P_DISABLE=1`) | working, special-purpose | granite-4.1-30b: 20.7 tok/s coherent ([details](../findings/rccl-dual-gpu-tp2.md)) |
