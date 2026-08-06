# Methodology

One box, one harness per round, every candidate on the same engine version.
This page describes the 2026-08 round (the "unified final"); round 1 differed
as noted in [`results/2026-05-round1.md`](results/2026-05-round1.md).

## Serving

- **Engine:** Ollama pinned at `ollama/ollama:0.32.5-rocm` for every
  candidate. Pinning mid-round is not optional — the 0.24 → 0.32.5 upgrade
  alone moved decode +30% on the eventual winner, which would have invalidated
  cross-candidate comparison.
- **Quantization:** GGUF q4_K_M throughout (a q8 probe in round 1 bought no
  measurable quality on these tasks). One candidate (mistral-small-4-119b) is
  a local `ollama create` import of unsloth's UD-Q4_K_M — the registry
  "mistral-small-4" at that time was a ~80B derivative masquerading as 119B
  (caught by size arithmetic: 47.8 GB blob ≠ 119B at q4).
- **Placement:** llama.cpp layer-split across 2× 48 GB; flash attention on,
  q8_0 KV cache (both verified working on gfx1100).

## Evaluation legs

1. **HumanEval-100** — the first 100 problems of HumanEval, identical set for
   all candidates (verified). Scored with the **official** `openai/human-eval`
   harness (vendored at master; the PyPI `human-eval` package is a third-party
   fork and was deliberately not used).
2. **12 synthetic tasks** — realistic prompts drawn from actual daily work
   (C#/.NET refactors, PowerShell tooling, pytest/numerics, resx/i18n,
   design questions). Objective legs measure latency/throughput; artifact
   quality was additionally hand-rated out-of-band.
3. **Agentic tool loop** — a multi-step rename-toolchain task requiring
   native structured tool calls, `MAX_TOOL_TURNS=12`. The metric is
   *convergence*: does the model ever return a turn free of tool calls
   (i.e. declare itself done)? This leg separated the field more than
   pass@1 did — 6 of 9 candidates never converged.

## Scoring hygiene

- **Sandboxed execution:** every generated solution ran under
  `docker run --rm --network none --cpus 6` with a single bind-mounted work
  dir. No model-generated code touched the host or network.
- **Normalization (applied uniformly, no per-candidate cases):** strip
  markdown fences; drop leading prose lines until code; drop
  `from __future__` lines (harness concatenation artifact — it prepends the
  prompt, so a restated module lands the future-import mid-file); guarantee
  column-0 start. Both raw and normalized scores were computed; normalization
  moved 6 of 9 candidates by 0.00 and never by more than +0.06 — it does not
  carry the ranking.
- **Deliberately not done:** de-indenting, trailing-prose trimming,
  stop-sequence truncation, repairing truncated code.

## The three-generation harness audit

The most transferable lesson of the round is that **the harness was wrong
twice before it was right**, in ways that specifically punish 2025/26
reasoning models. Generation 1 (512-token cap, `message.content` only) scored
gemma4:26b-a4b at 0.07 — its true score is 0.96. Full anatomy in
[`findings/thinking-models-vs-harnesses.md`](findings/thinking-models-vs-harnesses.md).
All published numbers come from the final generation (reasoning capture +
best-block extraction + HE 3072 / synth 4096 budgets), with every candidate
rerun on it. The superseded intermediate artifacts were kept, not averaged in.

## Threats to validity

- **n=1 hardware, n=1 run per leg.** Latency numbers are means over 100
  HumanEval generations plus single-shot synth legs; no cross-day variance
  study.
- **q4_K_M only.** Rankings could shift at other quant points (bf16 vLLM data
  exists only for a subset — see the findings).
- **HumanEval-100 saturates.** At 0.88–0.98 the pass@1 spread is small; the
  agentic-loop and speed columns carry most of the decision weight, which is
  a deliberate choice for the agentic-coding use case, not a general ranking.
- **Token caps still shape scores at the margin** even at 3072: one candidate
  (glm-4.5-air) remains budget-bound and is reported as a floor.
- **Synthetic tasks reflect one developer's workload** (heavy .NET/PowerShell/
  scientific-Python). They are not a public benchmark suite; their value here
  is identical conditions across candidates.
