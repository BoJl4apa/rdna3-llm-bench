# A card-pinned Ollama evicts on overflow; it does not spill to CPU

**TL;DR** — Ollama 0.35.0 (ROCm) was limited to one Radeon Pro W7800
(`ROCR_VISIBLE_DEVICES=1`, `HIP_VISIBLE_DEVICES=0`), then given a concurrent fan-out of three
~13–19 GB models at long contexts plus an embedding model, more than the card holds.
Its scheduler **evicted idle runners and reloaded**. Every runner left loaded
reported **100 % of its size in VRAM**, and no load offloaded layers to CPU. With 22.6 GB of the card
already held by another process (a vLLM pooling lane), the three models evicted
each other in turn, each paying a cold load (5–22 s). Pinning trades
co-residency for reload latency, not for a silent CPU slowdown.

## Why pin at all

Two reasons from the same box:

- An Ollama that sees both cards places a single-GPU model on the card with the
  most free memory. A resident audio stack on card 0 made card 1 the usual
  target, and the bias was observed going both ways.
- On a multi-GPU host, Ollama misreads a non-zero card's free memory as its
  **full total** once every card hosts a single-GPU runner, and stacks further
  models onto the already-occupied card. It was reported on CUDA with 0.32.15 and
  we confirmed it on ROCm with 0.35.0; as of 2026-10-11 the lookups are unchanged
  in v0.40.3 and on `main`. Upstream report:
  [ollama/ollama#18349](https://github.com/ollama/ollama/issues/18349), with our ROCm
  evidence in its comments; open fix:
  [ollama/ollama#18350](https://github.com/ollama/ollama/pull/18350). Pinning the
  main instance to one card removes the condition.

So the question was what pinning costs when a fan-out no longer fits one card:
**evict-and-reload, or silent CPU offload** (`size_vram < size`)?

## Method

A throwaway Ollama `0.35.0-rocm` container, card 1 only, reading the model store
read-only, with `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=6`, q8_0 KV cache
and flash attention. Three chat models were posted **concurrently** at their
production contexts (64 output tokens each), then an embedding model was loaded. For
each model we recorded `load_duration`, decode tok/s, the final `/api/ps` (`size`
against `size_vram`), and the scheduler's placement and eviction log lines.

## Run 1 — card 1 empty (0.5 GB used before)

| Model | ctx | Load (s) | tok/s | At the end (`/api/ps`) |
|---|---:|---:|---:|---|
| qwen3-coder:30b (30B-A3B) | 65536 | 7.5 | 103.9 | **evicted** |
| gemma4:26b-a4b | 65536 | 12.2 | 78.7 | 18.1 GB, 100 % VRAM |
| gpt-oss:20b | 32768 | 18.5 | 107.5 | 12.9 GB, 100 % VRAM |
| qwen3-embedding:4b | default | 1.1 | — | 7.1 GB, 100 % VRAM |

The scheduler logged `llama-server model predicted to exceed available memory,
evicting` with `predicted="19.3 GiB" available="23.5 GiB"`, then dropped the idle
`qwen3-coder` runner to load `gemma4`. The logged figures alone do not show the
overflow, and we did not trace the scheduler's decision rule; the eviction itself
is not in doubt. The three chat models do **not** co-reside
on one 48 GB card at these contexts.

## Run 2 — card 1 already holding 22.6 GB (a vLLM pooling lane)

| Model | ctx | Load (s) | tok/s | At the end (`/api/ps`) |
|---|---:|---:|---:|---|
| gemma4:26b-a4b | 65536 | 5.3 | 79.5 | **evicted** |
| qwen3-coder:30b | 65536 | 14.7 | 88.1 | **evicted** |
| gpt-oss:20b | 32768 | 22.0 | 104.2 | 12.9 GB, 100 % VRAM |
| qwen3-embedding:4b | default | 1.5 | — | 7.1 GB, 100 % VRAM |

There were two evictions: predicted 20.3 GiB against 4.0 GiB available, then 13.9 GiB against 1.1 GiB.
Each chat model displaced the one before it. Every load printed `offloaded N/N
layers to GPU`. Card 1 went from 22.6 GB to 43.8 GB used. The other card and its resident
workloads were untouched in both runs.

## What this means

- On one card, Ollama 0.35's scheduler prefers eviction over partial offload when
  a model will not fit next to the resident set. That is the good failure:
  reloads are visible as latency (`load_duration`), and decode speed stays at
  full-GPU rates.
- The price of pinning is set by your model mix. Here, a three-model concurrent
  call pays one reload with the card empty and three cold loads beside a 22 GB
  neighbour.

## Caveats

- n=1 box, gfx1100, ROCm 7.x, Ollama 0.35.0 only. Load times are from a warm NVMe
  page cache in some cases and a cold one in others: compare them within a run, not
  across runs.
- One fan-out shape (three chat models plus one embed). A single model larger than
  the card was not tested; eviction cannot help there.
- The accounting defect above is the reason to pin, not something this
  experiment measures.
