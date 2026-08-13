# 2026-08 addendum — Muse Glimmer 30B

Meta Superintelligence Labs released **Muse Glimmer 30B** (dense, thinking-mode,
vision-capable, Apache 2.0) on 2026-08-10 with the claim that it leads its size
class on agentic and coding work. We benched it three days later on the same
harness as the [unified final](2026-08-unified-final.md), against a fresh run
of the reigning candidate.

**Engine note.** Muse Glimmer support on AMD landed in Ollama 0.32.8, so this
addendum runs on `ollama/ollama:0.32.9-rocm` where the unified final used
0.32.5. To keep the August table comparable we re-ran the incumbent on the new
engine: pass@1 0.98 → 0.99 (run-to-run noise), decode 93.6 → 97.0 tok/s (+3%).
Read the two tables together with that drift in mind.

## Results

| candidate | params | pass@1 | decode tok/s | mean time-to-first-content (HE) | agentic tool loop |
|---|---|---:|---:|---:|---|
| **qwen3-coder:30b** (rerun, 0.32.9) | 30B-A3B MoE | **0.99** | **97.0** | 0.27 s | **converged** |
| muse-glimmer:30b | 30B dense | 0.97 | 27.5 | 33.3 s | 12-turn cap |

Both ran GGUF q4_K_M on a single 48 GB card (Muse Glimmer: 17 GB resident,
100% GPU), native structured tool calls, HumanEval cap 3072 / synth cap 4096,
vendor-recommended temperatures (0.2 vs 1.0 — see
[methodology](../methodology.md) on why we don't normalize).

Muse Glimmer detail: 2/100 empty completions, 4/100 capped at 3072, mean
~3.5K chars of chain-of-thought per HumanEval problem. Raw per-leg data (26
rows) in [`2026-08-muse-glimmer-per-leg.csv`](2026-08-muse-glimmer-per-leg.csv).

## Reading the result

- **The quality claim holds up.** 0.97 at temperature 1.0 is the best dense
  score we've measured on this box, level with 80B–120B MoEs, two points off
  the champion. For a 17 GB single-card footprint, that's a real size-class win.
- **The shape is wrong for interactive agentic serving.** Dense 30B on
  bandwidth-bound GDDR6 is 27.5 tok/s — 3.5× slower than a 3B-active MoE of
  equal quality — and its always-on thinking phase means ~33 s of silence
  before the first visible token on a typical coding prompt. The unified
  final's conclusion (*active-parameter bytes set the speed ceiling*) survives
  its strongest challenger yet.
- **"Agentic" didn't show in the loop.** On our multi-turn rename task it
  issued exactly one tool call per turn, explored serially, applied 2 of the
  4 required edits, and hit the 12-turn cap without declaring itself done —
  same non-convergence pattern as most of the August field. The incumbent
  remains the only fast model here that finishes the loop.

## Caveats

- Muse Glimmer supports a system-prompt reasoning-effort control
  (`Reasoning strength: low|medium|high|xhigh`). We benched at template
  default — the production-faithful config, since agentic harnesses inject
  their own system prompts — so its ceiling at `high`/`xhigh` is **untested**
  here. Expect the pass@1 gap to close at even lower speed; the throughput
  physics won't move.
- Vision input (its perception encoder) is out of scope for this harness.
- n=1 run per candidate, three days after model release, on a 3-day-old
  ROCm support path. Early numbers.
