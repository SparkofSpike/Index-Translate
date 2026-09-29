#!/usr/bin/env python3
"""Benchmark Index-Echo-S2ST dubbing pipeline backend variants.

Variants (one per process invocation; the model package directory is NEVER
modified — all patches are applied in-memory at runtime):

  baseline      stock pipeline, exactly as shipped
  vllm          CosyVoice3 LM token decode via a vLLM engine
                (post-load `cv.model.load_vllm(export_dir)`; the one-time
                export goes to a sidecar dir, not the package)
  compile       ST LM `generate()` via torch.compile(mode="reduce-overhead")
                + static KV cache
  vllm+compile  both patches

Methodology mirrors REPORT.md (2026-09-29, A100 80G): zh→en, clips of
1/3/6/10 s cut from the package's input_zh.wav, 1 warmup + N timed runs per
clip (mean reported); wall = extract()+synth() end-to-end; stlm = extract()
(ST LM decode + hidden states), tts = synth() (mapper + CV3 LM + flow + hift);
peak = torch.cuda.max_memory_allocated per run.

Usage:
    python bench_dub.py --model-dir /root/models/dubbing_2b_fulldir_cv3 --mode baseline
    python bench_dub.py --model-dir /root/models/dubbing_2b_fulldir_cv3 --mode vllm \
        --vllm-export /root/models/s2st_2b_vllm_export
    python bench_dub.py --model-dir /root/models/dubbing_2b_fulldir_cv3 --mode vllm+compile

Output: JSON (default bench_<mode>.json) + human-readable table on stdout.
The JSON keeps the raw transcript/translation of run 0 per clip for quality
eyeballing (all variants should produce identical text).

Requires: the model package's own runtime deps, plus `vllm` for the vllm
modes and a triton-enabled torch for the compile modes.
"""

import argparse
import json
import os
import sys
import time

import torch


def cut_clips(model_dir, seconds_list, workdir):
    """Cut the package's input_zh.wav into N-second clips (mono, native sr)."""
    import soundfile as sf

    src = os.path.join(model_dir, "input_zh.wav")
    assert os.path.isfile(src), f"{src} not found (expected inside the model package)"
    wav, sr = sf.read(src)
    if wav.ndim > 1:
        wav = wav[:, 0]
    os.makedirs(workdir, exist_ok=True)
    paths = {}
    for sec in seconds_list:
        p = os.path.join(workdir, f"clip_{sec}s.wav")
        sf.write(p, wav[: int(sec * sr)], sr)
        paths[sec] = p
    return paths


def apply_vllm_patch(pipe, export_dir):
    """Route CosyVoice3 LM decode through a vLLM engine (in-memory only).

    Mirrors what CosyVoice3(load_vllm=True) does, but with the export aimed at
    a writable sidecar dir instead of <pkg>/cosyvoice3/vllm. After this, the
    pipeline's existing `llm.inference_wrapper(...)` call in synth() picks the
    vLLM path automatically (it checks `hasattr(self, 'vllm')`).
    """
    import vllm  # noqa: F401  (fail fast with a clear ImportError)
    t0 = time.time()
    pipe.cv.model.load_vllm(export_dir)  # export is skipped if export_dir exists
    print(f"[patch] CosyVoice3 vLLM engine ready ({time.time() - t0:.1f}s), "
          f"export dir: {export_dir}", flush=True)


def apply_compile_patch(pipe):
    """Compile the ST LM decode path (in-memory only).

    Static KV cache makes generate() CUDA-graph friendly; reduce-overhead
    captures the decode step into graphs after warmup.
    """
    stlm = pipe.stlm
    stlm.llm.generation_config.cache_implementation = "static"
    stlm.llm.forward = torch.compile(stlm.llm.forward, mode="reduce-overhead")
    print("[patch] ST LM forward compiled (reduce-overhead, static cache)", flush=True)


def run_mode(model_dir, mode, clips, runs, warmup, lang, vllm_export, out_wav_dir):
    sys.path.insert(0, model_dir)
    os.chdir(model_dir)  # package internals use relative paths
    from modeling_dubbing import DubbingBridgeModel

    print(f"[load] {model_dir} ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    model = DubbingBridgeModel.from_pretrained(model_dir)
    load_peak_gb = torch.cuda.max_memory_allocated() / 2**30
    pipe = model._pipe

    if "vllm" in mode:
        apply_vllm_patch(pipe, vllm_export)
    if "compile" in mode:
        apply_compile_patch(pipe)

    results = []
    for sec, clip in clips.items():
        rec = {"clip_s": sec, "runs": []}
        for i in range(warmup + runs):
            torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            ext = pipe.extract(clip, lang=lang)
            t2 = time.perf_counter()
            out_wav = os.path.join(out_wav_dir, f"{mode}_{sec}s_r{i}.wav") if i == warmup else None
            w, info = pipe.synth(ext, out_wav=out_wav)
            t3 = time.perf_counter()
            entry = {
                "wall_s": round(t3 - t1, 3),
                "stlm_s": round(t2 - t1, 3),
                "tts_s": round(t3 - t2, 3),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "n_tokens": info["n_tokens"],
                "hit_eos": info["hit_eos"],
            }
            tag = "warmup" if i < warmup else f"run{i - warmup}"
            print(f"  [{mode}] {sec}s {tag}: wall={entry['wall_s']}s "
                  f"stlm={entry['stlm_s']}s tts={entry['tts_s']}s "
                  f"peak={entry['peak_gb']}G tokens={entry['n_tokens']}", flush=True)
            if i >= warmup:
                rec["runs"].append(entry)
            if i == 0:
                rec["src_text"] = ext["zh"]
                rec["tgt_raw"] = ext["tgt_raw"]
        for k in ("wall_s", "stlm_s", "tts_s", "peak_gb"):
            rec[f"mean_{k}"] = round(sum(r[k] for r in rec["runs"]) / len(rec["runs"]), 3)
        rec["mean_rtf"] = round(rec["mean_wall_s"] / sec, 3)
        results.append(rec)
    return {"mode": mode, "model_dir": model_dir, "lang": lang,
            "load_peak_gb": round(load_peak_gb, 2),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "results": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", "-m", required=True, help="dubbing_*_fulldir_cv3 package dir (read-only)")
    ap.add_argument("--mode", required=True, choices=["baseline", "vllm", "compile", "vllm+compile"])
    ap.add_argument("--lang", default="en", choices=["en", "es", "ja", "zh"])
    ap.add_argument("--clips", type=float, nargs="+", default=[1, 3, 6, 10])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--vllm-export", default=None,
                    help="sidecar dir for the one-time vLLM export "
                         "(default: <model-dir name>../<name>_vllm_export)")
    ap.add_argument("--out", default=None, help="JSON output path (default: bench_<mode>.json)")
    ap.add_argument("--workdir", default="/tmp/bench_dub_work", help="scratch dir for clips and wavs")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    vllm_export = args.vllm_export or os.path.join(
        os.path.dirname(model_dir), os.path.basename(model_dir) + "_vllm_export")
    os.makedirs(args.workdir, exist_ok=True)

    clips = cut_clips(model_dir, args.clips, args.workdir)
    data = run_mode(model_dir, args.mode, clips, args.runs, args.warmup,
                    args.lang, vllm_export, args.workdir)

    out = args.out or f"bench_{args.mode}.json"
    with open(out, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")
    print(f"{'clip':>5} {'wall':>7} {'stlm':>7} {'tts':>7} {'peak':>6} {'RTF':>5}")
    for r in data["results"]:
        print(f"{r['clip_s']:>5}s {r['mean_wall_s']:>7} {r['mean_stlm_s']:>7} "
              f"{r['mean_tts_s']:>7} {r['mean_peak_gb']:>5}G {r['mean_rtf']:>5}")


if __name__ == "__main__":
    main()
