# Thinking models break naive eval harnesses — a measured anatomy

**TL;DR** — 2025/26 reasoning-mode models emit chain-of-thought in a
*separate* API field (`message.thinking` in Ollama; `reasoning_content` in
OpenAI-compatible gateways) and leave `content` **empty until reasoning
closes**. A harness that reads only `content` under a small token cap records
a *blank reply* — not a truncated one — and scores the model at zero. Our
first scoring pass measured gemma4:26b-a4b at **0.07**; its true score on the
corrected harness is **0.96**. If your eval pipeline predates reasoning
models, your reasoning-model numbers may be floors.

## The three bugs, in the order we found them

### 1. Reasoning lands in a field you're not reading

The runner accumulated only `message.content` from the streaming response.
Thinking models spent the entire 512-token HumanEval budget inside
`message.thinking` and were cut off before emitting a single content token.
The evidence was exact, not statistical: **every single empty completion was
an exact-cap truncation** — 77/77 for gemma4:26b-a4b, 69/69 for
glm-4.7-flash, 57/57 for glm-4.5-air. The runner logged ~500 completion
tokens alongside an empty string: the tokens existed, in the other field.

### 2. Verbosity + small caps measures prose habits, not code

gpt-oss-120b (a non-hidden-CoT but very verbose model) hit the 512 cap on
94/100 problems — routinely mid-docstring, yielding
`unterminated triple-quoted string literal`. The 6 problems that fit all
passed. Its 0.35 measured documentation style. At a 3072 cap it scored 0.94.

### 3. "Take the first code fence" extracts the wrong block

Verbose models write explanatory snippets *before* the solution. A
first-fence extractor harvested those snippets and discarded the real answer
(cost gpt-oss-120b 16 completions on the intermediate rerun). The fix is a
best-block heuristic: among all fenced blocks, prefer ones containing a
function definition, then take the longest.

## The deltas (same models, same hardware, only the harness fixed)

| model | broken harness | corrected harness |
|---|---:|---:|
| gemma4:26b-a4b | 0.07 | 0.96 |
| glm-4.7-flash | 0.13 | 0.90 |
| gpt-oss-120b | 0.35 | 0.94 |
| glm-4.5-air | 0.25 | 0.63 (still budget-bound — a floor) |

Every jump is a harness artifact, not a model change. Note glm-4.5-air: some
models reason so long that even 3072 tokens is a cap — it needs 8–12k budgets
to measure at all. There is no universal "safe" cap; there is only checking
the empty-completion count.

## Checklist for eval authors

1. **Capture the reasoning field** (`message.thinking` /
   `reasoning_content`) even if you only score `content` — you need it to
   distinguish "model failed" from "model was still thinking."
2. **Budget ≥4k tokens** for any thinking-capable model, and report the
   capped/empty counts alongside pass@1.
3. **Treat empty completions as a harness alarm**, not a zero. Exact
   correlation between empties and cap-hits is the signature.
4. **Extract the best code block, not the first one.**
5. If you operate an OpenAI-compatible gateway, consider **server-side
   `max_tokens` floors (≥4096) on thinking-model routes** — small caps
   produce confusing blank replies for every downstream client, not just
   evals.
6. Corollary from our round 1: **don't disable thinking to save tokens** —
   gpt-oss:20b dropped 92.5% → 59% pass@1 with thinking off. The reasoning
   *is* the capability; budget for it instead.
