# Tensor parallelism on dual RDNA3: two RCCL defects, two workarounds, working TP=2

**TL;DR** — vLLM TP=2 across two root-port-separated RDNA3 cards (2× W7800,
PCIe Gen4 x8/x8, no P2P bridge) fails out of the box in two *independent*
ways: silent output corruption and a ~7000×-slow collective transport. Both
were root-caused and worked around; the result is coherent TP=2 bf16 at
20.7 tok/s on a 30B model. Filed upstream as
[ROCm/ROCm#6565](https://github.com/ROCm/ROCm/issues/6565) (corruption) and
[ROCm/ROCm#6576](https://github.com/ROCm/ROCm/issues/6576) (transport), with
the vLLM-side trail in
[vllm-project/vllm#38587](https://github.com/vllm-project/vllm/issues/38587).

## Defect 1 — silent all_gather corruption (ROCm#6565)

**Symptom:** TP=2 inference runs "successfully" and emits garbage tokens.
No error, no warning.

**Root cause:** during communicator setup (`devCommSetup`), RCCL issues an
async host-to-device copy of the 8-byte `devRingUserRanks` array and
synchronizes the stream; on this platform the sync reports completion while
the payload has not landed. Kernels then read zeros, every rank's ring-offset
collapses to slot 0, and all_gather returns rank-0 data in every slot.
Proven by D2H readback immediately after the "completed" copy: the device
buffer still holds zeros.

**Fix (zero runtime cost):** a verify-retry loop in `devCommSetup` — read the
copy back, retry until it sticks. Applied to RCCL 2.28.3 (`develop` branch),
built *inside* the vLLM ROCm container for ABI compatibility, injected via
`LD_PRELOAD`. Setup-time-only code path; steady-state throughput is
unaffected (20.7 tok/s vs. the 21.9 tok/s corrupted baseline).

**Diagnostic detour worth knowing:** `HSA_DISABLE_CACHE=1` also makes TP=2
coherent — it maps all VRAM uncached (MTYPE_UC on gfx11), which masks the
coherence bug but **halves compute throughput** (10.3 tok/s). Useful as a
one-line confirmation that you're hitting this class of bug; not a fix.

**Status:** still present in RCCL 2.30.4 (ROCm 7.14 `rdna` image, 12/12
failing with the identical signature). The patched-lib recipe remains
necessary on both 7.13 and 7.14 images.

## Defect 2 — P2P/IPC transport ~7000× slow (ROCm#6576)

**Symptom:** with correctness fixed, every collective stalls ~42 s per
inference step.

**Root cause isolation:** a 13-configuration matrix (process co-tenancy,
spawn-vs-fork, `HSA_USERPTR_FOR_PAGED_MEM`, `HSA_ENABLE_INTERRUPT`,
`HSA_USE_SVM`, transparent hugepages, …) produced **42.43 s to four
significant figures in every configuration** — until
`NCCL_P2P_DISABLE=1` dropped it to **0.029 s** (1477×), matching TP=1
behavior. `NCCL_DMABUF_ENABLE=0` alone does nothing: it is the P2P transport
itself, not the dmabuf attach path. The cost behaves like a ~215 ms fixed
penalty per collective on this dual-root-port topology, where the cards have
no direct P2P path and traffic must cross the CPU root complex.

**IOMMU exonerated:** rebooting with `amd_iommu=off` (zero IOMMU groups)
changes nothing for either defect. (RCCL's "Missing iommu=pt" warning greps
the kernel cmdline only to print itself; it gates no behavior.)

## The working recipe

```
LD_PRELOAD=/path/to/patched/librccl.so.1.0   # RCCL 2.28.3 develop + verify-retry patch
NCCL_P2P_DISABLE=1                           # force SHM transport; P2P is the stall
```

granite-4.1-30b, TP=2, bf16: **coherent, 20.72 tok/s**. Notes:

- Build the patched RCCL *inside* the target vLLM container — its toolchain
  works (amdllvm under `lib/llvm/bin/`), and container-ABI compatibility is
  what makes `LD_PRELOAD` safe.
- On dual-GPU RDNA3, treat TP=2 as a special-purpose lane (large-model batch
  work). Layer-split via llama.cpp/Ollama needs no collectives at all, pools
  the same 96 GB, and is the better production default.
- Single-GPU vLLM (TP=1) needs none of this — no RCCL in the path.

## Reproduction

Both upstream issues carry standalone reproducers and full numbers. The
corruption reproducer is a short PyTorch-distributed all_gather ground-truth
check; the transport issue reproduces with any TP=2 vLLM serve on a
dual-root-port RDNA3 pair.
