# Hebrew and Russian TTS on RDNA3: one worker, two queues

2026-11-13. Bounded GPU qualification followed by a shared Hebrew/Russian
serving worker. A native listener approved the selected Hebrew and Russian
audition voices for personal use; this is not a broad language-quality score.
The existing English Kokoro service was kept available throughout.

## BlueTTS bilingual candidate

BlueTTS 2.5 can synthesize Hebrew and Russian in one process on a Radeon Pro
W7800 (48 GB, gfx1100). With English Kokoro and two Whisper processes already
resident on the same GPU, it uses the remaining two compute queues, bringing
the total to **8/8**. Total card VRAM was about **20.0 GiB**. After a coordinated
load test, the fully loaded card returned to **15–17 W / 0% busy**.

| Test on the shared audio GPU | Complete synthesis | Audio duration |
|---|---:|---:|
| First Hebrew sentence | 0.620 s | 2.508 s |
| Different Hebrew sentence | 0.297 s | 4.319 s |
| Three Russian inputs | 0.216–0.241 s | 2.508–3.692 s |

These are complete synchronous synthesis times, not streaming first-audio or
client latency. Hebrew includes warm pronunciation preprocessing; Russian
starts with precomputed IPA. Its separately timed warm frontend took
0.074–0.103 s. Model loading is outside the table.

A second run repeated five fixtures ten times (50 syntheses), overlapping
three rounds of English STT, punctuation cleanup and English narration.
All requests completed and existing-service checks passed. HE/RU median
synthesis was 0.279/0.204 s, maximum 0.720/0.410 s. The worker remained loaded
for 45 seconds afterward and exited cleanly. This is one machine and a small,
repeated fixture set; it is not a multilingual quality benchmark or a soak test.

Raw sanitized evidence: [multilingual-tts-20261113.json](../results/multilingual-tts-20261113.json).
ASR recovered one Hebrew and two Russian fixtures apart from punctuation and
whitespace. The final listener-approved Russian choice below is Silero, not this BlueTTS RU candidate.

## Runtime and two traps

The working probe used Python 3.12, ONNX Runtime ROCm 1.21.0 from AMD's
**ROCm 6.4.4** wheel index, and system ROCm 6.4.4-129 inside the pinned
`ghcr.io/remsky/kokoro-fastapi-rocm:v0.10.0` image
(`sha256:53ef42fec42f173aab83d9750a7cad3ec899e8050c61b1e5ff6e670004fa93cc`).
The host stayed on kernel 6.8.0-149 and ROCm 7.2.3. Only the chosen render node
and KFD were exposed; `GPU_MAX_HW_QUEUES=1` was set. No host upgrade was needed.

1. **The wheel loaded ROCm SMI twice.** Its provider depended on bundled
   `librocm_smi64-a53a9aba.so.7.7.60404`; HIP also loaded the matching system
   `librocm_smi64.so.7.7.60404`. Both appeared in the process map. Synthesis
   succeeded, but process exit repeatedly failed with heap corruption, even
   with profiling disabled. Replacing only the provider's NEEDED reference
   with `librocm_smi64.so.7` eliminated the duplicate and the observed exit
   failure. Three subsequent runs exited cleanly, including the shared-card
   load test. Preserve the original binary and requalify other wheel versions;
   this is not a generic instruction to rewrite system libraries.
2. **A registered GPU provider need not be used.** In a separate ROCm 10.1 /
   MIGraphX 2.18 / ORT 1.29 probe, selecting the plugin by name silently made
   CPU-only sessions. Explicit device attachment fixed selection, as described
   in [ORT #33556](https://github.com/microsoft/onnxruntime/issues/33556).
   However, the first synthesis produced no audio in a seven-minute bounded
   compilation trial, so that trial was stopped. This does not establish that
   MIGraphX cannot compile the model under another configuration.

The successful direct-ROCm profiler placed all observed Conv, FusedConv, Gemm
and MatMul work on GPU; small shape/control operations stayed on CPU. The
legacy provider is discontinued upstream, so this is a compatibility result,
not a long-term upgrade strategy.

## Model, voice and frontends

- [BlueTTS runtime](https://github.com/maxmelichov/BlueTTS), commit
  `712f7c9491ad7875fc985e24e166afe43ae649b0`.
- [BlueTTS 2.5 ONNX bundle](https://huggingface.co/notmax123/BlueTTS2.5-onnx),
  revision `95feb5d0cbfc8e5a95bffb067d03537cbf35be3e`. Export manifest hash
  prefixes matched the downloaded binary/config files. Runtime/model metadata
  is MIT; always retain the matched vocabulary and latent statistics.
- Voice `libri_female_6147`: Liberty Stump, LibriTTS-R train-clean-100,
  CC BY 4.0 (Google LLC). 16 steps, CFG 3.0, speed 1.0; peak-limited 44.1 kHz
  mono PCM16. The bundle's separately listed in-house voice was not selected.
- Hebrew: RenikudPlus model revision
  `679c56ca449d41873fb8ff7711ddf7563d28198f`, CPU preprocessing. Its optional
  datastore was removed upstream; explicit `datastore=None` avoids a startup
  download failure.
- Russian: RUAccent turbo3.1/tiny mode plus RUPhon big and IPA remapping.
  These dependencies carry non-commercial licences: the whole Russian
  pipeline must not be called MIT. The model card's advertised Russian module
  was missing from the published bundle; the evaluator supplied an adapter,
  including a fix for mismatched tokenizer input/segment lengths.

## Selected serving pair: BlueTTS Hebrew + Silero Russian

The selected Russian model is [Silero v5_6_ru](https://github.com/snakers4/silero-models),
voice `xenia`, 48 kHz PCM16. Its CC BY-NC 4.0 weights constrain this deployment
to personal use. Hebrew retains BlueTTS 2.5 / `libri_female_6147`. RUAccent and
RUPhon are not dependencies of the selected production pair.

Both models stay resident in one worker on the audio GPU, alongside English
Kokoro and both Whisper services. PyTorch must initialize HIP **before** ORT
creates sessions: ORT-first selected a different HIP copy and PyTorch then
reported no GPU. With the ordered initialization and single-SMI packaging,
the pair shares two compute queues and exits cleanly.

The final HTTP worker completed 40 alternating HE/RU generations (two synthetic
fixtures repeated 20 times), overlapping three rounds of English STT, cleanup
and narration. All passed. With models still resident afterward, the audio
card held **8/8 queues, 20.6 GiB total VRAM, 14–18 W and 0% busy**. The other
card retained seven queues with coding/embedding models loaded. These values
are whole-card observations, not the worker's isolated memory/power cost.

| Selected worker request | Complete response | Returned audio |
|---|---:|---:|
| First HE sentence | 1.19 s | 4.32 s |
| First RU sentence | 1.48 s | 3.84 s |
| Warm HE number fixture | 0.30 s | 3.41 s |
| Warm RU number fixture | 0.033 s | 3.89 s |
| Approximately 500 HE characters | 2.31 s | 37.55 s |
| Approximately 500 RU characters | 0.126 s | 30.48 s |

The worker limits inputs/output, allows one active generation, rejects excess
work with 429, and cancels between 100-character chunks. A real TCP disconnect
released the worker and completed a subsequent short HE request in 0.43 s.
Speed changes use pitch-preserving ffmpeg tempo conversion. The initial RU
number test exposed silent digit omission in raw Silero input; deterministic
number expansion fixed it. ASR recovered both languages' numbers, an English
word embedded in Hebrew and long passages' final sentences. Repeated-sentence
ASR can collapse repetitions, so it is not a phoneme-level correctness oracle.

Evidence: [selected worker measurements](../results/multilingual-serving-20261113.json).
This is a small bounded qualification on one machine, not endurance testing.
Mixed Russian/English and Russian/Hebrew text is explicitly rejected rather
than partially spoken. Desktop/phone language routing and Stop acceptance
remain separate client tests. Warm residency and low idle power passed;
broader mixed-language pronunciation remains future work.
