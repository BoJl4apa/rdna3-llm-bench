# Tool-loop "convergence" is a per-run draw, not a model property

**TL;DR:** the agentic-tool-loop column in our 2026-08 results was measured at
**n=1 per candidate**. Re-measuring the winner at n=18–25 per session shows the
outcome is **bimodal run-to-run** and unstable **across sessions** — the same
model, prompt, temperature and engine produced anywhere from 1/20 to 24/25
"converged" runs on different days. A controlled A/B against the leading
environmental suspect (a sibling container pinning a GPU at 100% busy) came
back **negative**. Single-draw convergence cells — ours included — should not
be trusted, and we've reworked the harness so they can't be produced again.

## The two modes

`qwen3-coder:30b` (30B-A3B MoE, q4_K_M, Ollama /api/chat native tool calls,
temp 0.2) on a fixed 12-site rename task over stubbed `grep` / `read_file` /
`edit_file` / `list_files` tools, `MAX_TOOL_TURNS=12`:

- **Parallel-batch mode:** the model emits 16–21 tool calls batched across
  ≤5 turns, finishes all 12 edit sites, and returns a text-only "done" turn.
  This is the "converged, 4 turns" the 2026-08 table recorded.
- **Serial mode:** the model emits exactly one tool call per turn, works
  steadily through the sites, and hits the 12-turn cap mid-task.

Which mode a run lands in looks like a sampling event at the first tool turn,
not a capability. Both modes come from the same weights at the same settings,
back-to-back.

**A structural observation that took us too long to see:** a 12-site rename
needs at least 14 serial turns (1 search + 12 edits + the closing answer), so
a 12-turn cap **guarantees** serial-mode runs are scored as failures however
diligently they work. With per-run task scoring (below), capped serial runs
show 8/12 sites edited — steady progress, not wandering. If your harness caps
turns below the serial-path length of the task, your "convergence" column is
measuring *batching propensity*, not task ability.

## Session history (same model / prompt / temp / engine)

| session (2026) | box state | converged | note |
|---|---|---|---|
| 09-01 a | sibling GPU queue stuck at 100% busy | 4/18 | pre-instrumentation |
| 09-01 b | quiet | **24/25** | pre-instrumentation; the outlier |
| 09-02 a | quiet, gate-verified | 1/5 | |
| 09-02 b | quiet, gate-verified | 1/20 | controlled arm A |
| 09-02 c | stuck queue reproduced, verified | 1/20 | controlled arm B |

Five of six sessions cluster at ~5–25%; one measured 96% and predates the
box-state instrumentation, so its conditions are unrecoverable. We flag it
rather than discard it: it is the strongest evidence that something
session-scoped (still unidentified) can move this number massively.

## The confound experiment: stuck GPU queue → refuted

Our whisper.cpp/HIP containers had a bug where an uncapped hardware-queue
config leaves a card at 100% `gfx_busy` / ~80–120 W indefinitely after use
(see the [idle-power finding](resident-model-idle-power-vulkan-vs-hip.md)).
The 4/18 session ran with that spin live; the 24/25 session ran after the fix
— an alarming correlation ("does a background container's stuck queue perturb
a *different process's* sampling?") worth a controlled test:

| arm (n=20 each) | converged | task complete (12/12 sites) | decode tok/s |
|---|---|---|---|
| A — quiet (0% busy, 4 W idle, verified) | 1/20 | 3/20 | 94.7 |
| B — spin live (100% busy / 78–104 W on the sibling card, verified) | 1/20 | 1/20 | 93.9 |

**Identical convergence, ≤1% decode delta.** On this topology (model on one
card, spin on the other), a sibling container's stuck queue costs power, not
model behaviour. The 24/25 outlier remains unexplained — and is exactly why
the harness now records box state with every run.

## What we changed in the harness

1. **Repeats:** tool-use prompts run N times per candidate; results are
   reported as **counts over n** (`1/20`), never a bare rate — a percentage
   from n=5 invites the same over-reading that a single draw does.
2. **`exit_reason` per run** (`answered` / `capped` / `error` /
   `no_tool_calls`): "converged" = answered after ≥1 executed tool call. A
   model that never calls a tool, or errors out, no longer counts as
   converged; an answer on the final allowed turn no longer counts as capped.
3. **Deterministic task scoring:** the stub world has exactly 12 known rename
   sites, so each run is scored on `sites_edited` (correct old→new per site)
   with `task_complete` = 12/12 — **separately from convergence**. A run that
   converges in 4 turns having edited 2 sites is a fast non-worker; a capped
   run at 8/12 is a slow worker. The single "converged" bit hid exactly this.
4. **Quiet-box gate:** the runner snapshots per-card busy/power, resident
   models, and engine version at start and end of every candidate, and
   refuses to start on a busy box unless explicitly overridden.

## Takeaways if you bench agentic loops

- Never publish a single-draw convergence cell. Ours said "converged, 4
  turns"; the underlying long-run behaviour was ~1-in-5 at best.
- Score task completion and loop termination as **separate** metrics; either
  alone misleads.
- Size the turn cap to the task's serial path, or say explicitly that you are
  measuring call-batching propensity.
- Record box state with every measurement. The one session we can't explain
  is the one we didn't instrument.

*Measured 2026-09-01 → 2026-09-02, Ollama 0.33.2-rocm, single W7800 per
model, q4_K_M, num_ctx 16384. The 2026-08 round column ran on 0.32.5–0.33.x
during engine drift; all repeat sessions above are on 0.33.2.*
