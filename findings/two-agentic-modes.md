# "Agentic" is two skills, and they don't correlate

**TL;DR** — We scored six models on two multi-turn tool-use scenarios that
differ in *where the information comes from*. In one, the work is enumerable
up front: fan out independent tool calls, done. In the other — an iterative
fix→observe→fix loop — the next step only exists after you act: edit, re-run a
fail-fast suite, read the new failure, edit again. The rankings barely overlap.
The incumbent daily driver leads the first (4/5) and is the **only** model of
six that fails the second (1/5); three models that never finish the first drive
the second to green 5/5; one model does both. A single-scenario agentic score
measures one of these and silently generalizes to the other.

> **Naming, because we've used "two modes" before.** A
> [previous finding](tool-loop-convergence-instability.md) used it for two
> behaviours *within one model on one task* — batching many tool calls per turn
> versus emitting one per turn. This page is about two **task shapes**, and how
> models rank differently across them. The earlier result is a load-bearing
> input here, not a duplicate: batching-vs-serial is exactly what our first
> scenario ends up measuring.

## The scenarios

Both run over a small canned repo behind four tools — `grep`, `read_file`,
`edit_file` (exact-match replace), `list_files` — via native structured tool
calls, `MAX_TOOL_TURNS=12`, **n=5 runs per model per scenario**, temperature
0.2, Ollama `0.33.2-rocm`, GGUF q4_K_M.

**A. `legacy-rename-toolchain` — enumerable up front.** Rename a symbol across
12 known sites. Everything is discoverable from one `grep`; nothing is learned
by executing anything. Complete = all 12 sites edited.

**This scenario measures batching propensity, and does so by construction.** A
strictly serial run needs ≥13 turns (one search, 12 edits, plus a closing
answer) against a 12-turn cap, so a model that emits one tool call per turn is
scored as failing however diligently it works. We knew that going in (it is the
[previous finding's](tool-loop-convergence-instability.md) structural point)
and left the scenario unchanged for cross-round comparability. Read column A as
*calls per turn*, not as "can this model rename 12 sites".

**B. `staged-rate-limit-fix` — revealed by acting.** Two defects in a rate
limiter, and `run_tests` is **fail-fast**, so the second defect is *invisible*
until the first is fixed. Complete = both defects fixed **and** the suite
re-run to green. A fix without re-running doesn't count; neither does a green
claim the harness never saw.

**C. `null-shipment-diagnosis` — read-only control.** A `NullReferenceException`
whose throw site is a red herring; the root cause is a builder that never
assigns the field. Edits are refused and counted. The answer must end in two
structured lines naming file and member, so scoring parses an answer instead of
grading prose.

## The result

`task_complete` runs out of 5.

| model | params | pass@1 | tok/s | A. rename | B. fix-loop | C. diagnosis |
|---|---|---:|---:|:---:|:---:|:---:|
| qwen3-coder:30b | 30B-A3B MoE | 0.99 | 96.7 | **4/5** | 1/5 | 5/5 |
| qwen3.8:27b⁰ | 27.3B dense | 0.98 | 42.1 | **5/5** | 5/5 | 5/5 |
| qwen3.5-122b | 122B MoE | 0.97 | 41.8 | 0/5 | 5/5 | 5/5 |
| gemma4:26b-a4b | 26B-A4B MoE | 0.93 | 73.9 | 0/5 | 5/5 | 5/5 |
| nemotron-3.5-lightning | 32.9B-A3B MoE | 0.87 | **161** | 0/5 | 5/5 | 5/5 |
| laguna-xs-2.1 | 33.4B | 0.62¹ | 90.5 | 0/5 | 3/5 | 5/5 |

⁰ The `qwen3.8:27b` tag is digest-identical to `qwen3.8:27b-mtp-q4_K_M`, so this
row is the MTP build. The plain-q4 arm scores the same on all three scenarios
([A/B](../results/2026-09-qwen3.8.md)), so nothing in this finding turns on it.

¹ A token-budget floor, not a capability measurement: 1.00 on the 58 problems
it finishes under the cap. See the [round page](../results/2026-09-round.md).

The two columns are **independent, not opposed**. `qwen3.8:27b` has both
skills. `qwen3-coder:30b` has only the first. Three models have only the
second. That is the whole argument for measuring them separately: no ordering
of models reproduces both columns.

Column C separates **nobody** — six for six. It is a fair task (the cause never
appears in the stack trace, and a plausible-looking null check sits in an
unrelated file), and every model walks it. We publish it as a null result and
keep it as a control: a scenario can be multi-turn, tool-driven and genuinely
agentic in shape while carrying no signal at all.

## Column A is a batching measurement — here is the evidence

Tool calls issued per run in scenario A, against a hard 12-turn cap:

| model | tool calls per run | sites edited | reading |
|---|---|---|---|
| qwen3.8:27b | 37, 54, 38, 42, 38 | 12,12,12,12,12 | heavy batcher, ~3–4.5 calls/turn |
| qwen3.5-122b | 24, 25, 35, 26, 34 | 6, 7, 11, 7, 11 | batches, doesn't finish in 12 turns |
| qwen3-coder:30b | 20, 19, 19, 12, 20 | 12,12,12, 4,12 | batches ~1.7 calls/turn |
| nemotron-3.5-lightning | 12, 12, 12, 12, 12 | 3, 2, 10, 3, 2 | **strictly serial** — one call per turn |
| gemma4:26b-a4b | 12, 12, 7, 12, 12 | 10, 0, 4, 4, 0 | strictly serial |
| laguna-xs-2.1 | 12, 12, 12, 12, 12 | 2, 2, 0, 0, 0 | strictly serial |

Three models emit **exactly one tool call per turn** — 12 of them whenever they
run to the cap, which is 14 of their 15 runs. They are bounded by the cap, not
by the task. Their 0/5 says "does not batch",
and nothing about whether they could rename 12 sites given 14 turns. Column A
is honest about what it measures once you read it that way; it is dishonest if
you label it "agentic ability", which is what our 2026-08 round did.

## Why B ranks differently

Scenario B is immune to the batching shortcut: you cannot fan out the second
fix, because the second defect does not exist in the observable world until the
first one lands. Every model must run, observe, and revise. Five of six do it.

The incumbent's five runs, in detail:

| run | exit | defect A | defect B | tests run | outcome |
|---|---|---|---|---|---|
| 1 | capped | fixed | not found | 2 | incomplete |
| 2 | **no tool calls** | — | — | 0 | answered in prose, touched nothing |
| 3 | **no tool calls** | — | — | 0 | answered in prose, touched nothing |
| 4 | answered | fixed | fixed | 3 | complete |
| 5 | capped | fixed | not found | 2 | incomplete |

Two distinct failures, neither of them a bad edit — **no model in the field
made a single wrong edit in this scenario**. In two runs it never engaged the
tools at all, replying with an analysis instead of an action. In two more it
fixed defect A, re-ran the suite, and ground to the cap without locating
defect B. Compare nemotron-3.5-lightning: 7–10 tool calls, 2 edits, 2–3 test
runs, green, 5/5 — the minimum viable loop, every time.

We are not claiming batchers are bad at loops in general; `qwen3.8:27b`
falsifies that outright. The claim is narrower and better supported: **these are
separate capabilities, and one popular agentic model has one and not the
other.**

## A second trap: completing the task and declaring done are different metrics

`qwen3.8:27b` edited all 12 rename sites in **5/5** runs and returned a
tool-call-free turn in **1/5**. It does the job, then keeps working. Our
2026-08 round scored this leg as *convergence* ("did the model ever declare
itself done?"), which would have ranked it 1/5 — a false negative on a model
that succeeded every time. `laguna-xs-2.1` shows the same split in scenario B:
two of its three completions ran to the cap after the suite was already green.

We now record both, per run: `task_complete_runs` (did the work land) and
`tool_converged_runs` (did the loop terminate), plus an `exit_reason` of
`answered` / `capped` / `error` / `no_tool_calls`. A run can converge without
completing and complete without converging. One number cannot say which
happened.

## A third: the answer format is part of the scenario

Scenario C originally ended open ("find the root cause") and was scored by a
keyword lexicon over free prose. That version rejected valid paraphrases and
credited red-herring words the prompt itself had supplied. Rewritten to demand
two structured lines — `ROOT_CAUSE_FILE:` and `ROOT_CAUSE_MEMBER:` — and scored
by parsing them, the *same models on the same engine* moved: the weakest
candidate went from 2/5 with three runs hitting the turn cap to 5/5 with none.

Two things changed at once and both are worth naming. The scorer stopped
guessing at meaning. And the prompt acquired a **termination criterion** — an
open-ended "diagnose this" gives a weak model no signal that it is finished, so
it keeps calling tools until the cap. If your agentic scenario asks for prose,
part of what you are measuring is the model's willingness to stop talking.

## And the prerequisite: repeat, or you are flipping coins

Everything above is n=5 because n=1 on this metric is not a measurement — the
same model, prompt, temperature and engine produced 1/20 to 24/25 convergence
across sessions, and the environmental confound we suspected was refuted by a
controlled A/B. That is a separate writeup:
[tool-loop convergence is a per-run draw](tool-loop-convergence-instability.md).
Its conclusion is the precondition for this one.

## What we'd tell someone building an agentic eval

1. **Ship at least two scenario shapes** — one enumerable up front, one where
   the world only reveals the next step in response to an action.
2. **Make the second stateful and fail-fast.** If edits don't really apply and
   the test tool doesn't really hide the second failure, you have written a
   batching task with extra turns.
3. **Check your cap against the serial path length** of the task, or your
   headline column is measuring calls-per-turn without saying so.
4. **Score the task, not the termination.** Neither implies the other.
5. **Repeat, and publish counts over n.**
6. **Consider routing by loop shape.** We did: the batching model stays the
   daily driver, and the fastest reliable loop-closer became a second route for
   repair-loop work. Its pass@1 is 12 points lower — for that loop, quality was
   not the binding constraint.

## Caveats

- **Canned worlds, not a real repo.** Tools operate on in-memory files, and
  `run_tests` uses a string oracle over the defect signatures rather than
  executing anything. A model that deletes the offending line reads as
  "fixed". These scenarios discriminate; they do not certify.
- **n=5 per model per scenario.** Enough to separate 1/5 from 5/5, not enough
  to rank 4/5 against 5/5.
- **One engine version, one quant, one temperature, one 12-turn cap.** The cap
  is a scoring parameter, and scenario A is especially sensitive to it.
- **Six models, one box.** This is a claim about these six under these
  scenarios, not a law about model families.
