# vLLM's native W4A16 kernels on RDNA3: 7.4× single-stream, 125 tok/s aggregate

**TL;DR** — vLLM v0.23.0 ships native RDNA3 int4 kernels
([#41394](https://github.com/vllm-project/vllm/pull/41394) dense,
[#44075](https://github.com/vllm-project/vllm/pull/44075) fused-MoE). On a
W7800 (gfx1100) they take the *same model file* from 3.9 to **29.3 tok/s**
single-stream, and to **125.8 tok/s aggregate at 8-way concurrency with 71 ms
TTFT**. Quantized vLLM serving on RDNA3 goes from dead to competitive — with
sharp edges documented below. As of testing (2026-08) there was near-zero
independent field validation of these kernels; this page is that datum.

## A/B (same model, same box, kernel engagement verified in logs)

Model: gemma-3-27b GPTQ, symmetric int4, group_size=128 (compressed-tensors).

| | vLLM 0.23.0 native kernel | vLLM 0.19.1 Triton fallback |
|---|---|---|
| single-stream decode | **29.3 tok/s** | 3.9 tok/s |
| 8-way aggregate | **125.8 tok/s** (71 ms TTFT) | not measured |

The aggregate number is the headline for multi-consumer serving: llama.cpp's
batching cannot match it on this hardware.

## The gates (get these wrong and you silently lose 7×)

1. **Symmetric GPTQ int4 only** for the dense path. An AWQ-asymmetric control
   ran 3.1 tok/s on the *new* image — it silently falls back to the old
   Triton path. No warning; check your throughput.
2. **group_size=128** is the validated configuration; gs=32/64 had no
   matching kernel on RDNA3 in our earlier testing.
3. The MoE path takes 4-bit compressed-tensors MoE checkpoints.
4. gfx1100 must be in the image's compiled arch list (the
   `rocm/vllm:...rdna...` image tags have it).

## Operational landmines

- **`HSA_OVERRIDE_GFX_VERSION=11.0.0` breaks ROCm 7.14 images** — torch init
  dies ("No CUDA GPUs are available"). The override is a habit from older
  stacks; native gfx1100 support makes it unnecessary. Keep it for 7.13-era
  images, drop it for 7.14+.
- **Long context is not fully validated:** attention still runs a Triton
  fallback path;
  [vllm#50603](https://github.com/vllm-project/vllm/issues/50603) reports
  long-context corruption on gfx1100 + ROCm 7.14. Clean in our runs to
  ~12K context; treat 32K+ as unverified and check outputs.
- TP=1 needs no RCCL and sidesteps the
  [dual-GPU RCCL defects](rccl-dual-gpu-tp2.md) entirely. TP=2 on the 7.14
  image still needs both workarounds from that page.

## Practical read for RDNA3 owners

If you serve one interactive stream, llama.cpp/Ollama GGUF remains simpler
and (for MoE models) faster. The W4A16 lane earns its complexity when you
need *concurrency* — many parallel agents/tenants against one dense model —
where 8-way aggregate throughput and sub-100 ms TTFT are out of llama.cpp's
reach. Publishable symmetric-GPTQ quants of current models are scarce;
self-quantization may be required for the model you actually want.
