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
2. **ST LM `torch.compile` / CUDA graph** — the ST LM decode goes through HF
   `generate()`; compiling `llm.forward` with `mode="reduce-overhead"` and a
   static KV cache captures the decode step into CUDA graphs.

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

## Output

`bench_<mode>.json` + a stdout table:

```
clip    wall    stlm     tts   peak   RTF
   1s   4.36    ...      ...   8.3G  4.36
```
