# S2ST Backend Speedup Bench

English | [中文](README_zh.md)

Follow-up experiments for the two directions listed in the 2026-09-29 single-GPU
benchmark report (`REPORT.md` §5):

1. **CosyVoice3 LM decode on vLLM** — the package ships the switch
   (`CosyVoice3(load_vllm=...)`, currently hardcoded off in
   `code/tts/pipeline.py`). The pipeline's `synth()` generates speech tokens via
   `llm.inference_wrapper(...)`, which already routes to a vLLM engine when one
   is attached — so the whole experiment is a **runtime-only** patch, no change
   to the model package.
2. **ST LM `torch.compile`** — the ST LM decode goes through HF
   `generate()`; compiling `llm.forward` speeds up the decode loop
   (`mode="default"` — CUDA-graph modes are incompatible with this
   architecture, see Results caveats).

`bench_dub.py` measures both, per mode, with the same methodology as the
original report (zh→en, 1/3/6/10 s clips from the package's `input_zh.wav`,
1 warmup + 3 timed runs, wall / stlm / tts / peak-VRAM).

## Usage

```bash
# baseline (reference, reproduces the original numbers)
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode baseline

# CosyVoice3 LM on vLLM (one-time export to a sidecar dir, package untouched)
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode vllm \
    --vllm-export /path/to/s2st_2b_vllm_export

# ST LM torch.compile
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode compile

# both
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode vllm+compile
```

One mode per process — the vLLM patch frees the torch copy of the LM layers
(`del ...layers`), so modes are not combinable within a single run beyond the
provided `vllm+compile` combination.

Notes:

- **Never benchmark on a GPU that serves production** (e.g. the box running the
  online S2ST/S2TT workers). Use an idle card, and never stop an online
  process to free VRAM — queue or move to another machine instead.
- TensorRT (`load_trt`) is intentionally **not** covered: the package does not
  ship the flow-decoder `.plan` (GPU-specific), so TRT needs an on-box
  ONNX→TRT export first. vLLM is the cheap 80% of that direction.
- The vLLM engine is created with `gpu_memory_utilization=0.2` (upstream
  default in `CosyVoice2Model.load_vllm`) — about 16 GB extra on an A100 80G.
- JSON output keeps run-0 transcript/translation per clip so output text can
  be diffed against baseline for a quality sanity check.

## Results (2026-09-29, A100 80G, `dubbing_2b_fulldir_cv3`)

Isolated venv: py3.12 + torch 2.13+cu129 + vllm 0.29.0+cu129; baseline/vllm
runs on transformers 5.6.0 (same as production), compile runs on
transformers 5.17.0 (see caveats below). RTF = wall / clip duration,
lower is better. All four modes produce **identical transcripts** on every
clip (quality-neutral).

| clip | baseline | vllm | compile | vllm+compile |
|---|---|---|---|---|
| 1s | 4.52 | 2.94 | 2.99 | **1.05** |
| 3s | 1.72 | 1.00 | 0.98 | **0.44** |
| 6s | 1.18 | 0.69 | 0.81 | **0.35** |
| 10s | 0.68 | 0.42 | 0.50 | **0.19** |

Stage split at 10s (wall seconds, mean): ST LM 2.47 → 2.44 (vllm) →
0.39 (compile) → 0.36 (both); TTS 4.28 → 1.72 (vllm) → 4.57 (compile) →
1.54 (both). Peak VRAM in vllm mode *dropped* 8.2G → 6.8G.

**Combined speedup: 3.6× end-to-end RTF at 10s** — well past the 30% target
in the original report.

### Caveats (ruled-out paths, kept for the record)

- `mode="reduce-overhead"` + static KV cache does **not** work on this
  hybrid linear-attention architecture: `KeyError: linear_attention` on
  transformers 5.6 (no masking-utils entry for the hybrid arch), and an
  inductor segfault (`conv_state.copy_` vs cudagraphs) on 5.17. The
  compile numbers above use **`mode="default"`** (no CUDA graphs), which
  is stable and captures most of the win (~7× on the ST LM stage).
- vLLM runs need a one-line environment patch: the vllm 0.29 sampler
  constant `MAX_NUM_STOP_TOKEN_IDS=128` is below CosyVoice3's 200 stop
  tokens; raised to 256 in the bench venv (tensor widths derive from this
  constant, kernels adapt by shape — safe). Not an upstream code change.

## Output

`bench_<mode>.json` + a stdout table:

```
clip    wall    stlm     tts   peak   RTF
   1s   4.36    ...      ...   8.3G  4.36
```
