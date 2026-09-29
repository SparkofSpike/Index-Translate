# S2ST 后端提速基准

[English](README.md) | 中文

对应 2026-09-29 单卡基准报告（`REPORT.md` §5）列的两个提速方向的验证实验：

1. **CosyVoice3 LM 解码换 vLLM 后端**——包里自带开关
   （`CosyVoice3(load_vllm=...)`，当前在 `code/tts/pipeline.py` 里写死关闭）。
   管线的 `synth()` 走 `llm.inference_wrapper(...)` 生成语音 token，该方法在
   挂载 vLLM 引擎后会自动切换路径——所以整个实验是**纯运行时补丁**，
   不改动模型包任何文件。
2. **ST LM `torch.compile` / CUDA graph**——ST LM 解码走 HF `generate()`；
   用 `mode="reduce-overhead"` 编译 `llm.forward` 并配静态 KV cache，
   可把 decode 步捕获进 CUDA graph。

`bench_dub.py` 按与原始报告相同的方法论分别测量各模式（zh→en，
从包内 `input_zh.wav` 裁剪 1/3/6/10s 四档，预热 1 次 + 计时 3 次取均值，
指标含 wall / stlm / tts / 峰值显存）。

## 用法

```bash
# 基线（对照，复现原始数据）
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode baseline

# CosyVoice3 LM 走 vLLM（一次性导出到旁路目录，模型包保持只读）
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode vllm \
    --vllm-export /path/to/s2st_2b_vllm_export

# ST LM torch.compile
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode compile

# 两者叠加
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode vllm+compile
```

每个进程只跑一种模式——vLLM 补丁会释放 torch 侧 LM 层
（`del ...layers`），除提供的 `vllm+compile` 组合外不可在单进程内切换。

说明：

- TensorRT（`load_trt`）**暂不覆盖**：包内未附带 flow decoder 的
  `.plan`（GPU 相关产物），需要先在本机做 ONNX→TRT 导出。vLLM 是该方向
  性价比最高的 80%。
- vLLM 引擎按上游默认 `gpu_memory_utilization=0.2` 创建——A100 80G 上约
  额外占 16 GB。
- JSON 输出保留了每个 clip 第 0 次的转写/译文文本，可与 baseline 逐字
  对比做质量抽检。

## 输出

`bench_<mode>.json` + stdout 表格：

```
clip    wall    stlm     tts   peak   RTF
   1s   4.36    ...      ...   8.3G  4.36
```
